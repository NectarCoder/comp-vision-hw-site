# TODO: Add description
from __future__ import annotations

# Segments objects using ArUco markers as an auto-generated GrabCut guide

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# Tracks segmentation progress and log output for this module
LOGGER = logging.getLogger("module3_part2")


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARUCO_DICTIONARY_NAMES: Tuple[str, ...] = (
	"DICT_6X6_250",
	"DICT_6X6_100",
	"DICT_5X5_100",
	"DICT_5X5_50",
	"DICT_4X4_250",
	"DICT_4X4_100",
	"DICT_4X4_50",
	"DICT_ARUCO_ORIGINAL",
)


@dataclass
class DetectionResult:
	hull: np.ndarray
	ids: np.ndarray


# Builds the CLI so users can choose folders and tuning knobs
def _build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Segment an object boundary using ArUco markers as anchors.")
	parser.add_argument("image_folder", type=Path, help="Folder containing the input images (non-recursive).")
	parser.add_argument("--suffix", default="_detect", help="Suffix appended to output filenames (default: _detect).")
	parser.add_argument("--iterations", type=int, default=5, help="Number of GrabCut refinement iterations (default: 5).")
	parser.add_argument("--hull-expansion", type=float, default=0.15, help="Radial hull expansion fraction (default: 0.15).")
	parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity.")
	return parser


# Configures stdout logging noise level based on the flag
def _setup_logging(verbose: bool) -> None:
	logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")


# Creates detector parameters that work across OpenCV releases
def _get_detector_parameters(aruco_module):
	if hasattr(aruco_module, "DetectorParameters"):
		params_cls = aruco_module.DetectorParameters
		return params_cls.create() if hasattr(params_cls, "create") and callable(params_cls.create) else params_cls()
	if hasattr(aruco_module, "DetectorParameters_create"):
		return aruco_module.DetectorParameters_create()
	raise AttributeError("Could not instantiate ArUco DetectorParameters; update OpenCV.")


# Yields every dictionary constant your OpenCV build supports
def _iter_available_dictionaries(aruco_module) -> Iterable[int]:
	for dict_name in ARUCO_DICTIONARY_NAMES:
		if hasattr(aruco_module, dict_name):
			yield getattr(aruco_module, dict_name)


# Runs the detector API that matches the user's OpenCV version
def _detect_with_dictionary(gray: np.ndarray, dictionary, parameters, aruco_module):
	if hasattr(aruco_module, "ArucoDetector"):
		detector = aruco_module.ArucoDetector(dictionary, parameters)
		return detector.detectMarkers(gray)
	return aruco_module.detectMarkers(gray, dictionary, parameters=parameters)


def _detect_aruco_markers(image: np.ndarray) -> DetectionResult | None:
	if not hasattr(cv2, "aruco"):
		LOGGER.error("OpenCV is missing the aruco module; please install opencv-contrib-python.")
		return None

	aruco_module = cv2.aruco
	parameters = _get_detector_parameters(aruco_module)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	best_corners: List[np.ndarray] | None = None
	best_ids: np.ndarray | None = None
	best_count = 0

	for dict_id in _iter_available_dictionaries(aruco_module):
		dictionary = aruco_module.getPredefinedDictionary(dict_id)
		corners, ids, _ = _detect_with_dictionary(gray, dictionary, parameters, aruco_module)
		marker_count = 0 if ids is None else len(corners)
		if marker_count > best_count:
			best_corners = corners
			best_ids = ids
			best_count = marker_count
		if marker_count >= 3:
			break

	if best_corners is None or best_ids is None or best_count < 3:
		LOGGER.warning("Detected fewer than 3 markers; skipping image (best=%s).", best_count)
		return None

	pts = np.vstack([c.reshape(-1, 2) for c in best_corners]).astype(np.float32)
	hull = cv2.convexHull(pts).reshape(-1, 2)
	return DetectionResult(hull=hull, ids=best_ids.reshape(-1))


# Scans for markers and returns the convex hull around all of them
def _detect_aruco_hull(image: np.ndarray) -> np.ndarray | None:
	detection = _detect_aruco_markers(image)
	return detection.hull if detection else None


# Expands the hull outward so GrabCut starts with a bit of padding
def _expand_hull(hull: np.ndarray, expansion: float, image_size: Tuple[int, int]) -> np.ndarray:
	if expansion <= 0:
		return hull
	centroid = hull.mean(axis=0, keepdims=True)
	expanded = centroid + (hull - centroid) * (1.0 + expansion)
	w, h = image_size
	expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
	expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
	return expanded


# Builds the GrabCut seed mask from the hull and some morphology tricks
def _build_grabcut_mask(image_shape: Tuple[int, int], hull: np.ndarray, expansion: float) -> np.ndarray:
	h, w = image_shape
	mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
	expanded_hull = _expand_hull(hull, expansion, (w, h))
	hull_int = np.clip(expanded_hull.astype(np.int32), [0, 0], [w - 1, h - 1])
	fg_seed = np.zeros((h, w), dtype=np.uint8)
	cv2.fillConvexPoly(fg_seed, hull_int, 1)
	kernel = np.ones((7, 7), np.uint8)
	sure_fg = cv2.erode(fg_seed, kernel, iterations=1)
	probable_fg = cv2.dilate(fg_seed, kernel, iterations=2)
	probable_bg = cv2.dilate(fg_seed, kernel, iterations=6)
	mask[probable_bg == 0] = cv2.GC_BGD
	mask[probable_fg == 1] = cv2.GC_PR_FGD
	mask[sure_fg == 1] = cv2.GC_FGD
	return mask


# Runs GrabCut and returns a smoothed binary segmentation
def _run_grabcut(image: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
	bgd_model = np.zeros((1, 65), dtype=np.float64)
	fgd_model = np.zeros((1, 65), dtype=np.float64)
	cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
	result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
	return cv2.medianBlur(result, 5)


# Finds the largest contour so we have a single clean boundary
def _extract_primary_contour(binary_mask: np.ndarray) -> np.ndarray | None:
	contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return max(contours, key=cv2.contourArea) if contours else None


# Draws the extracted contour plus a translucent fill on top of the source
def _draw_boundary(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
	overlay = image.copy()
	cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=3)
	mask = np.zeros(image.shape[:2], dtype=np.uint8)
	cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
	colored_mask = cv2.merge([mask // 3, mask, np.zeros_like(mask)])
	blended = cv2.addWeighted(image, 0.75, colored_mask, 0.25, 0)
	return cv2.addWeighted(blended, 0.7, overlay, 0.3, 0)


# Iterates supported images in sorted order while skipping previous outputs
def _iter_image_files(folder: Path, skip_suffix: str | None = None) -> Iterable[Path]:
	for path in sorted(folder.iterdir()):
		if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
			continue
		if skip_suffix and path.stem.endswith(skip_suffix):
			continue
		yield path


# Produces the output filename alongside the original
def _output_path(input_path: Path, suffix: str) -> Path:
	return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


# Handles the full marker detection + GrabCut workflow for one image
def _process_single_image(image_path: Path, suffix: str, iterations: int, expansion: float) -> bool:
	LOGGER.info("Processing %s", image_path.name)
	image = cv2.imread(str(image_path))
	if image is None:
		LOGGER.error("Failed to read %s", image_path)
		return False

	hull = _detect_aruco_hull(image)
	if hull is None:
		return False

	mask = _build_grabcut_mask(image.shape[:2], hull, expansion)
	segmentation = _run_grabcut(image, mask, iterations)
	contour = _extract_primary_contour(segmentation)
	if contour is None:
		LOGGER.warning("No contour could be extracted for %s", image_path.name)
		return False

	visualized = _draw_boundary(image, contour)
	output_file = _output_path(image_path, suffix)
	if cv2.imwrite(str(output_file), visualized):
		LOGGER.info("Saved %s", output_file.name)
		return True

	LOGGER.error("Could not write result to %s", output_file)
	return False


# Drives the batch pass over the entire folder and reports success
def _run(image_folder: Path, suffix: str, iterations: int, expansion: float) -> int:
	if not image_folder.exists() or not image_folder.is_dir():
		LOGGER.error("%s is not a valid folder", image_folder)
		return 1

	image_paths = list(_iter_image_files(image_folder, skip_suffix=suffix))
	if not image_paths:
		LOGGER.error("No supported image files were found in %s", image_folder)
		return 1

	successes = 0
	for image_path in image_paths:
		try:
			if _process_single_image(image_path, suffix, iterations, expansion):
				successes += 1
		except cv2.error as exc:
			LOGGER.exception("OpenCV error while processing %s: %s", image_path.name, exc)
		except Exception as exc:  # noqa: BLE001 - log unexpected errors explicitly
			LOGGER.exception("Unexpected error while processing %s: %s", image_path.name, exc)

	if successes == 0:
		LOGGER.warning("No files were successfully processed.")
		return 2

	LOGGER.info("Completed %s/%s files.", successes, len(image_paths))
	return 0


# Entry point for CLI usage
def main(argv: Sequence[str] | None = None) -> int:
	parser = _build_argument_parser()
	args = parser.parse_args(argv)
	_setup_logging(args.verbose)
	return _run(args.image_folder, args.suffix, args.iterations, args.hull_expansion)


if __name__ == "__main__":
	sys.exit(main())
