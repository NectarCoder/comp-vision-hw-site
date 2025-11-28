from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def open_video_capture(path: str):
    return cv2.VideoCapture(path)


def validate_input_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path)) if Path(path).is_file() else None


def ensure_writer(output_path: Optional[str], fps: float, w: int, h: int) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    resolved = str(output_path)
    suffix = Path(resolved).suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v"}:
        codec_candidates = ["avc1", "H264", "mp4v"]
    elif suffix in {".avi"}:
        codec_candidates = ["XVID", "mp4v"]
    else:
        codec_candidates = ["mp4v"]

    for code in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(resolved, fourcc, fps, (w, h))
        if writer and writer.isOpened():
            print(f"[module5_6_part2] Using codec {code} for output {resolved}")
            return writer
        if writer:
            writer.release()
            writer = None
            print(f"[module5_6_part2] Failed to open VideoWriter with codec {code}, trying next candidate…")

    print(f"[module5_6_part2] Could not initialize a VideoWriter for {resolved}.")
    return None


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.25):
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    overlay = frame.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)
    frame[:] = overlay

    ys, xs = np.where(mask_bool)
    if ys.size and xs.size:
        cv2.rectangle(frame, (xs.min(), ys.min()), (xs.max(), ys.max()), color, 2)


def _match_mask_to_frame(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == frame_shape:
        return mask.astype(bool) if mask.dtype != bool else mask
    resized = cv2.resize(mask.astype(np.uint8), (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def render_masks(cap: cv2.VideoCapture, writer: Optional[cv2.VideoWriter], masks: np.ndarray, show_window: bool = False):
    if masks.ndim == 3 and masks.shape[-1] < min(masks.shape[0], masks.shape[1]):
        masks = np.moveaxis(masks, -1, 0)

    total_frames = masks.shape[0]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = masks[idx] if idx < total_frames else masks[-1]
        overlay_mask(frame, _match_mask_to_frame(mask, frame.shape[:2]))
        if writer:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()


def _resolve_checkpoint(checkpoint: Optional[str]) -> str:
    checkpoint_path = checkpoint or os.environ.get("SAM_CHECKPOINT")
    if checkpoint_path is None:
        for candidate in (
            Path.home() / ".cache" / "sam" / "sam_vit_b.pth",
            Path("hwsources/resources/m5_6/sam_vit_b.pth"),
        ):
            if Path(candidate).is_file():
                checkpoint_path = str(candidate)
                break
    if not checkpoint_path:
        raise RuntimeError("No SAM checkpoint found. Provide --checkpoint or set SAM_CHECKPOINT.")
    return checkpoint_path


def _select_model_key(checkpoint: str, registry_keys):
    stem = Path(checkpoint).stem.lower()
    for tag in ("vit_h", "vit_l", "vit_b"):
        if tag in stem:
            match = next((k for k in registry_keys if tag in k.lower()), None)
            if match:
                return match
    for prefer in ("vit_h", "vit_l", "vit_b"):
        match = next((k for k in registry_keys if prefer in k.lower()), None)
        if match:
            return match
    return next(iter(registry_keys))


def _load_sam_model(checkpoint: str, device: str):
    try:
        from segment_anything import sam_model_registry
    except Exception as exc:
        raise RuntimeError("segment_anything is not installed or could not be imported") from exc

    model_key = _select_model_key(checkpoint, sam_model_registry.keys())

    sam_model = sam_model_registry[model_key](checkpoint=checkpoint)
    try:
        sam_model.to({"mps": "mps", "cuda": "cuda"}.get(device, "cpu"))
    except Exception:
        pass
    return sam_model


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if not ys.size or not xs.size:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def generate_masks_with_sam(video_path: str, device: str = "cpu", resize_scale: float = 1.0, checkpoint: Optional[str] = None):
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor
    except Exception as exc:
        raise RuntimeError("segment_anything is not installed or could not be imported") from exc

    checkpoint_path = _resolve_checkpoint(checkpoint)
    sam_model = _load_sam_model(checkpoint_path, device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    predictor = SamPredictor(sam_model)

    cap = open_video_capture(video_path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Failed to open input {video_path}")

    MIN_AREA_RATIO = 0.002
    MIN_CONFIDENCE = 0.35
    MIN_IOU = 0.2
    MAX_TRACK_GAP = 5

    masks = []
    processed = 0
    last_mask_small: Optional[np.ndarray] = None
    miss_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = frame if resize_scale == 1.0 else cv2.resize(frame, (int(frame.shape[1] * resize_scale), int(frame.shape[0] * resize_scale)))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)

        candidate_mask: Optional[np.ndarray] = None
        candidate_score = 0.0
        candidate_area = 0.0

        if last_mask_small is not None:
            box = _mask_to_box(last_mask_small)
            if box is not None:
                try:
                    masks_pred, scores_pred, _ = predictor.predict(box=box[None, :], point_coords=None, point_labels=None, multimask_output=False)
                    candidate_mask = masks_pred[0].astype(bool)
                    candidate_score = float(scores_pred[0])
                    candidate_area = float(candidate_mask.sum())
                except Exception:
                    candidate_mask = None

        need_reacquire = candidate_mask is None or miss_counter >= MAX_TRACK_GAP
        if need_reacquire:
            try:
                masks_data = mask_generator.generate(frame_rgb)
            except Exception:
                masks_data = []
            if masks_data:
                segments = [entry["segmentation"].astype(bool) for entry in masks_data]
                areas = [seg.sum() for seg in segments]
                confidences = [float(entry.get("predicted_iou", entry.get("stability_score", 1.0))) for entry in masks_data]
                idx = int(np.argmax(areas))
                candidate_mask = segments[idx]
                candidate_area = float(areas[idx])
                candidate_score = confidences[idx]
                miss_counter = 0

        frame_area = small_frame.shape[0] * small_frame.shape[1]
        area_ratio = (candidate_area / frame_area) if frame_area and candidate_mask is not None else 0.0
        meets_area = area_ratio >= MIN_AREA_RATIO
        meets_conf = candidate_score >= MIN_CONFIDENCE
        meets_iou = True
        if candidate_mask is not None and last_mask_small is not None:
            inter = np.logical_and(candidate_mask, last_mask_small).sum()
            union = np.logical_or(candidate_mask, last_mask_small).sum()
            iou = (inter / union) if union else 0.0
            meets_iou = iou >= MIN_IOU

        if candidate_mask is not None and meets_area and meets_conf and meets_iou:
            stable_small = candidate_mask
            miss_counter = 0
        elif last_mask_small is not None and miss_counter < MAX_TRACK_GAP:
            stable_small = last_mask_small
            miss_counter += 1
        else:
            stable_small = np.zeros(small_frame.shape[:2], dtype=bool)
            miss_counter += 1
            if miss_counter >= MAX_TRACK_GAP:
                last_mask_small = None

        if stable_small.any():
            last_mask_small = stable_small.copy()

        mask_full = _match_mask_to_frame(stable_small, frame.shape[:2])
        masks.append(mask_full.astype(np.uint8))
        processed += 1
        if processed % 50 == 0:
            print(f"SAM tracker processed {processed} frames…")

    cap.release()
    if not masks:
        raise RuntimeError("No frames were processed from the input video.")
    return np.stack(masks, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Markerless SAM tracker (video-only)")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Device for SAM model (if available)")
    parser.add_argument("--no-window", action='store_true', help='Don\'t show GUI window (headless mode)')
    parser.add_argument("--resize", type=float, default=1.0, help="Scale frames for segmentation (0.0-1.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM checkpoint (overrides env var)")
    # Note: output NPZ and output video are automatically derived from the input path: <stem>-masks.npz and <stem>-tracked.mp4
    args = parser.parse_args(argv)

    input_path = args.input
    sample_default = Path("hwsources/resources/m5_6/aruco-marker.mp4")
    if not input_path and sample_default.is_file():
        input_path = str(sample_default)

    if not (0.0 < args.resize <= 1.0):
        print("--resize must be within (0.0, 1.0].")
        sys.exit(1)

    input_valid = validate_input_path(input_path)
    if not input_valid:
        print("Invalid or missing input path. Please pass a valid video file path.")
        sys.exit(1)
    input_path = input_valid

    cap = open_video_capture(input_path)
    if not cap or not cap.isOpened():
        print(f"Failed to open input {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # Build default output paths
    in_path = Path(input_path)
    default_masks = in_path.with_name(in_path.stem + "-masks.npz")
    default_out = in_path.with_name(in_path.stem + "-tracked.mp4")

    masks_path = default_masks
    output_path = default_out

    writer = ensure_writer(str(output_path), fps, w, h)
    if writer is None:
        raise RuntimeError(f"Unable to initialize video writer for {output_path}. Please install H.264/MP4 codecs.")

    try:
        print("Generating masks with SAM tracker…")
        masks = generate_masks_with_sam(input_path, device=args.device, resize_scale=args.resize, checkpoint=args.checkpoint)
    except Exception as exc:
        raise RuntimeError(f"SAM tracking failed: {exc}") from exc

    print(f"Saving masks NPZ to {masks_path}")
    np.savez_compressed(str(masks_path), masks=masks.astype(np.uint8))

    cap2 = open_video_capture(input_path)
    render_masks(cap2, writer, masks, show_window=not args.no_window)


if __name__ == "__main__":
    main()
