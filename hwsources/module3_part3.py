#TODO: add description

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import cv2
import numpy as np
import torch
import module3_part2 as aruco_module


LOGGER = logging.getLogger("module3_part3")


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sam2Runtime:
	predictor: Any
	device: str


def _build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Compare ArUco segmentation with SAM2 outputs for a folder of images.",
	)
	parser.add_argument(
		"image_folder",
		type=Path,
		help="Folder containing the evaluation images (non-recursive).",
	)
	parser.add_argument(
		"--suffix",
		default="_sam2",
		help="Suffix appended to the visualization output files (default: _sam2).",
	)
	parser.add_argument(
		"--aruco-iterations",
		type=int,
		default=5,
		help="Number of GrabCut iterations for the ArUco baseline (default: 5).",
	)
	parser.add_argument(
		"--aruco-hull-expansion",
		type=float,
		default=0.15,
		help="Fractional hull expansion reused from Part 2 (default: 0.15).",
	)
	parser.add_argument(
		"--bbox-scale",
		type=float,
		default=0.05,
		help="Extra padding applied to the SAM2 bounding box prompt (default: 0.05).",
	)
	parser.add_argument(
		"--prompt-points",
		type=int,
		default=4,
		help="Number of positive point prompts sampled from the hull (default: 4, 0 to disable).",
	)
	parser.add_argument(
		"--model-config",
		type=Path,
		required=True,
		help="Absolute or relative path to the SAM2 YAML config downloaded locally.",
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		required=True,
		help="Absolute or relative path to the SAM2 model weights (.pt) stored locally.",
	)
	parser.add_argument(
		"--device",
		default="auto",
		help="Torch device to use (cpu/cuda/mps). The default 'auto' picks CUDA when available.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable debug logging.",
	)
	return parser


def _setup_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(levelname)s: %(message)s",
	)


def _resolve_device(device_arg: str) -> str:
	if device_arg.lower() != "auto":
		return device_arg
	if torch is not None and torch.cuda.is_available():
		return "cuda"
	return "cpu"


def _load_sam2_predictor(config_path: Path, checkpoint_path: Path, device_arg: str) -> Sam2Runtime | None:
	try:
		from hydra.utils import instantiate  # type: ignore
		from omegaconf import OmegaConf  # type: ignore
		from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
	except ImportError:
		LOGGER.error(
			"SAM2 dependencies are missing. Install 'sam2', 'hydra-core', 'omegaconf', and 'torch' as listed in setup_venv/requirements.txt."
		)
		return None

	if torch is None:
		LOGGER.error("PyTorch is not installed; SAM2 cannot run without torch.")
		return None

	config_path = config_path.expanduser()
	checkpoint_path = checkpoint_path.expanduser()
	if not config_path.is_file():
		LOGGER.error("SAM2 config not found: %s", config_path)
		return None
	if not checkpoint_path.is_file():
		LOGGER.error("SAM2 checkpoint not found: %s", checkpoint_path)
		return None

	device = _resolve_device(device_arg)
	LOGGER.info("Loading SAM2 model (config=%s, checkpoint=%s, device=%s)", config_path, checkpoint_path, device)
	try:
		cfg = OmegaConf.load(str(config_path))
		OmegaConf.resolve(cfg)
		if "model" not in cfg:
			raise ValueError("Configuration file is missing the 'model' key required for instantiation.")
		model = instantiate(cfg["model"], _recursive_=True)
		state_dict = _load_checkpoint_state(checkpoint_path)
		missing_keys, unexpected_keys = model.load_state_dict(state_dict)
		if missing_keys:
			raise RuntimeError(f"Checkpoint is missing keys: {missing_keys}")
		if unexpected_keys:
			raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected_keys}")
		model = model.to(device)
		model.eval()
		predictor = SAM2ImagePredictor(model)
		return Sam2Runtime(predictor=predictor, device=device)
	except Exception as exc:  # noqa: BLE001 - bubble up helpful message
		LOGGER.exception("Failed to initialize SAM2: %s", exc)
		return None


def _load_checkpoint_state(checkpoint_path: Path) -> dict:
	load_kwargs = {"map_location": "cpu"}
	try:
		weights = torch.load(str(checkpoint_path), weights_only=True, **load_kwargs)  # type: ignore[arg-type]
	except TypeError:
		weights = torch.load(str(checkpoint_path), **load_kwargs)

	if isinstance(weights, dict):
		if "model" in weights and isinstance(weights["model"], dict):
			return weights["model"]
		return weights
	raise TypeError("Checkpoint did not contain a valid state dictionary.")


def _segment_with_aruco(
	image: np.ndarray,
	iterations: int,
	expansion: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	detection = aruco_module._detect_aruco_markers(image)
	if detection is None:
		LOGGER.warning("Skipping image because ArUco markers were not detected.")
		return None

	mask = aruco_module._build_grabcut_mask(image.shape[:2], detection.hull, expansion)
	segmentation = aruco_module._run_grabcut(image, mask, iterations)
	contour = aruco_module._extract_primary_contour(segmentation)
	if contour is None:
		LOGGER.warning("Could not extract a contour from the ArUco segmentation.")
		return None
	return detection.hull, segmentation, contour


def _compute_bounding_box(
	hull: np.ndarray,
	image_shape: Tuple[int, int],
	scale: float,
) -> np.ndarray:
	x_min = float(np.min(hull[:, 0]))
	y_min = float(np.min(hull[:, 1]))
	x_max = float(np.max(hull[:, 0]))
	y_max = float(np.max(hull[:, 1]))
	width = x_max - x_min
	height = y_max - y_min
	pad_x = width * max(scale, 0.0)
	pad_y = height * max(scale, 0.0)
	w, h = image_shape
	bbox = np.array(
		[
			max(0.0, x_min - pad_x),
			max(0.0, y_min - pad_y),
			min(w - 1.0, x_max + pad_x),
			min(h - 1.0, y_max + pad_y),
		],
		dtype=np.float32,
	)
	return bbox


def _build_prompt_points(hull: np.ndarray, count: int, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
	if count <= 0 or hull.size == 0:
		return None, None
	count = min(count, len(hull))
	indices = np.linspace(0, len(hull) - 1, num=count, dtype=int)
	selected = hull[indices]
	centroid = hull.mean(axis=0, keepdims=True)
	points = np.vstack([selected, centroid])
	w, h = image_shape
	points[:, 0] = np.clip(points[:, 0], 0, w - 1)
	points[:, 1] = np.clip(points[:, 1], 0, h - 1)
	labels = np.ones(len(points), dtype=np.int32)
	return points.astype(np.float32), labels.astype(np.int32)



def _segment_with_sam2(
	runtime: Sam2Runtime,
	image: np.ndarray,
	bbox: np.ndarray,
	hull: np.ndarray,
	aruco_mask: np.ndarray,
	prompt_points: int,
) -> Tuple[np.ndarray, float] | Tuple[None, None]:
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	runtime.predictor.set_image(image_rgb)
	point_coords, point_labels = _build_prompt_points(hull, prompt_points, image.shape[1::-1])
	predict_kwargs: dict[str, Any] = {"box": bbox, "multimask_output": True}
	if point_coords is not None and point_labels is not None:
		predict_kwargs["point_coords"] = point_coords
		predict_kwargs["point_labels"] = point_labels
	try:
		masks, scores, _logits = runtime.predictor.predict(**predict_kwargs)
	except Exception as exc:  # noqa: BLE001 - log and fail gracefully
		LOGGER.exception("SAM2 prediction failed: %s", exc)
		return None, None
	if masks is None or len(masks) == 0:
		return None, None
	masks_u8 = [(mask * 255).astype(np.uint8) for mask in masks]
	best_idx = 0
	best_iou = -1.0
	for idx, cand in enumerate(masks_u8):
		iou = _calculate_iou(aruco_mask, cand)
		if iou > best_iou:
			best_iou = iou
			best_idx = idx
	return masks_u8[best_idx], float(scores[best_idx] if scores is not None else 0.0)


def _calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
	a = mask_a > 0
	b = mask_b > 0
	union = np.logical_or(a, b).sum()
	if union == 0:
		return 0.0
	intersection = np.logical_and(a, b).sum()
	return float(intersection / union)


def _draw_mask_panel(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
	panel = image.copy()
	overlay = np.zeros_like(panel)
	overlay[mask > 0] = color
	panel = cv2.addWeighted(panel, 0.7, overlay, 0.3, 0)
	contour = aruco_module._extract_primary_contour(mask)
	if contour is not None:
		cv2.drawContours(panel, [contour], -1, color, 2)
	return panel


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
	labeled = panel.copy()
	cv2.putText(labeled, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
	return labeled


def _compose_visualization(
	image: np.ndarray,
	aruco_contour: np.ndarray,
	sam_mask: np.ndarray,
	aruco_mask: np.ndarray,
	iou: float,
) -> np.ndarray:
	aruco_panel = aruco_module._draw_boundary(image, aruco_contour)
	aruco_panel = _label_panel(aruco_panel, "ArUco GrabCut")
	sam_panel = _draw_mask_panel(image, sam_mask, (255, 0, 255))
	sam_panel = _label_panel(sam_panel, "SAM2")
	combined = image.copy()
	if aruco_contour is not None:
		cv2.drawContours(combined, [aruco_contour], -1, (0, 255, 0), 2)
	sam_contour = aruco_module._extract_primary_contour(sam_mask)
	if sam_contour is not None:
		cv2.drawContours(combined, [sam_contour], -1, (255, 0, 255), 2)
	text = f"IoU: {iou:.3f}"
	combined = _label_panel(combined, text)
	stacked = np.concatenate([aruco_panel, sam_panel, combined], axis=1)
	return stacked


def _iter_image_files(folder: Path, suffix: str) -> Iterable[Path]:
	for path in sorted(folder.iterdir()):
		if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
			continue
		if path.stem.endswith(suffix):
			continue
		yield path


def _output_path(input_path: Path, suffix: str) -> Path:
	return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _process_single_image(
	image_path: Path,
	runtime: Sam2Runtime,
	suffix: str,
	aruco_iterations: int,
	aruco_expansion: float,
	bbox_scale: float,
	prompt_points: int,
) -> bool:
	LOGGER.info("Processing %s", image_path.name)
	image = cv2.imread(str(image_path))
	if image is None:
		LOGGER.error("Failed to read %s", image_path)
		return False

	aruco_result = _segment_with_aruco(image, aruco_iterations, aruco_expansion)
	if aruco_result is None:
		return False
	hull, aruco_mask, aruco_contour = aruco_result
	bbox = _compute_bounding_box(hull, image.shape[1::-1], bbox_scale)
	sam_mask, sam_score = _segment_with_sam2(runtime, image, bbox, hull, aruco_mask, prompt_points)
	if sam_mask is None:
		LOGGER.warning("SAM2 failed to segment %s", image_path.name)
		return False

	iou = _calculate_iou(aruco_mask, sam_mask)
	visualization = _compose_visualization(image, aruco_contour, sam_mask, aruco_mask, iou)
	output_path = _output_path(image_path, suffix)
	if cv2.imwrite(str(output_path), visualization):
		LOGGER.info("Saved %s (IoU=%.3f, SAM-score=%.3f)", output_path.name, iou, sam_score or 0.0)
		return True
	LOGGER.error("Failed to persist result for %s", image_path.name)
	return False


def _run(
	image_folder: Path,
	suffix: str,
	aruco_iterations: int,
	aruco_expansion: float,
	bbox_scale: float,
	prompt_points: int,
	runtime: Sam2Runtime,
) -> int:
	if not image_folder.is_dir():
		LOGGER.error("%s is not a valid folder", image_folder)
		return 1
	image_paths = list(_iter_image_files(image_folder, suffix))
	if not image_paths:
		LOGGER.error("No supported image files were found in %s", image_folder)
		return 1

	successes = 0
	for image_path in image_paths:
		try:
			if _process_single_image(
				image_path,
				runtime,
				suffix,
				aruco_iterations,
				aruco_expansion,
				bbox_scale,
				prompt_points,
			):
				successes += 1
		except cv2.error as exc:
			LOGGER.exception("OpenCV error while processing %s: %s", image_path.name, exc)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("Unexpected error while processing %s: %s", image_path.name, exc)

	if successes == 0:
		LOGGER.warning("SAM2 pipeline failed for all files.")
		return 2
	LOGGER.info("Completed %s/%s files.", successes, len(image_paths))
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	parser = _build_argument_parser()
	args = parser.parse_args(argv)
	_setup_logging(args.verbose)
	runtime = _load_sam2_predictor(args.model_config, args.checkpoint, args.device)
	if runtime is None:
		return 1
	return _run(
		args.image_folder,
		args.suffix,
		args.aruco_iterations,
		args.aruco_hull_expansion,
		args.bbox_scale,
		args.prompt_points,
		runtime,
	)


if __name__ == "__main__":
	sys.exit(main())
