from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    from . import module2_part1 as m1
    from . import module2_part2 as m2
except ImportError: 
    import module2_part1 as m1  
    import module2_part2 as m2 


def find_template_files(folder: Path, limit: int = 10) -> List[Path]:
    # Verify the folder exists and is a directory
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Template folder not found: {folder}")
    # Get and sort all template files, limited to the first 10
    tpl_files = sorted([folder / f for f in os.listdir(folder) if f.startswith("template_")])
    return tpl_files[:limit]


def match_template_once(scene_gray: np.ndarray, tpl_gray: np.ndarray, threshold: float = 0.8):
    # Get template dimensions and scene dimensions
    th, tw = tpl_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]
    # Skip if template is larger than the scene
    if th > sh or tw > sw:
        return [], []

    # Perform template matching to find similarity scores
    res = cv2.matchTemplate(scene_gray, tpl_gray, cv2.TM_CCORR_NORMED)
    # Find all locations where the score exceeds the threshold
    y_idxs, x_idxs = np.where(res >= threshold)
    boxes = []
    scores = []
    # Convert pixel coordinates into bounding box format
    for (y, x) in zip(y_idxs, x_idxs):
        boxes.append((int(x), int(y), int(x + tw), int(y + th)))
        scores.append(float(res[y, x]))

    # Keep only the top candidates to save computation
    max_candidates = 200
    if len(scores) > max_candidates:
        order = np.argsort(scores)[::-1][:max_candidates]
        boxes = [boxes[i] for i in order]
        scores = [scores[i] for i in order]

    return boxes, scores


def draw_detections(image: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float, str]]) -> np.ndarray:
    # Create a copy of the image to draw on
    out = image.copy()
    # Draw a green box and label for each detection
    for (box, score, label) in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label}:{score:.2f}"
        cv2.putText(out, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return out


def blur_regions(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], blur_multiplier: float = 2.0) -> np.ndarray:
    # Create a copy of the image to modify
    out = image.copy()
    # Process each bounding box region to apply blur
    for (x1, y1, x2, y2) in boxes:
        # Ensure coordinates stay within image bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(out.shape[1], x2), min(out.shape[0], y2)
        # Skip if the region is invalid
        if x2c <= x1c or y2c <= y1c:
            continue
        # Extract the region to be blurred
        region = out[y1c:y2c, x1c:x2c]

        # Determine the kernel size and sigma based on region dimensions
        ksize, sigma = m2._resolve_kernel_params(region.shape, None, None)

        # Increase the blur strength by the multiplier factor
        sigma = float(sigma) * float(blur_multiplier)
        ksize = max(3, int(round(ksize * blur_multiplier)))

        # Ensure kernel size is odd (required by OpenCV)
        ksize = max(3, ksize)
        if ksize % 2 == 0:
            ksize += 1

        # Apply Gaussian blur to the region
        blurred_region = m2.apply_gaussian_blur(region, ksize, sigma)

        # Place the blurred region back into the output image
        out[y1c:y2c, x1c:x2c] = blurred_region

    return out


def main() -> None:
    # Get scene image and templates folder from command line or user input
    if len(sys.argv) == 3:
        scene_arg = sys.argv[1]
        tpl_folder_arg = sys.argv[2]
    elif len(sys.argv) == 1:
        scene_arg = input("Scene file path: ").strip()
        tpl_folder_arg = input("Templates folder path: ").strip()
    else:
        print("Usage: python module2_part3.py <scene_path> <templates_folder>")
        print("Or run without args to be prompted for paths.")
        return

    # Convert arguments to Path objects
    scene_path = Path(scene_arg)
    tpl_folder = Path(tpl_folder_arg)

    # Verify the scene file exists
    if not scene_path.exists() or not scene_path.is_file():
        print(f"Scene not found: {scene_path}")
        return

    # Get list of template files from the folder
    try:
        template_files = find_template_files(tpl_folder, limit=10)
    except Exception as e:
        print(f"Error finding templates: {e}")
        return

    # Check if any template files were found
    if not template_files:
        print(f"No templates found in {tpl_folder} (expecting files starting with 'template_').")
        return

    # Load the scene image and convert to grayscale
    scene = cv2.imread(str(scene_path))
    if scene is None:
        print(f"Failed to load scene image: {scene_path}")
        return
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # Initialize list to store all detected objects from all templates
    all_detections: List[Tuple[Tuple[int, int, int, int], float, str]] = []

    # Set matching threshold and blur strength parameters
    threshold = 0.75
    blur_multiplier = 2.5

    # Loop through each template and find matches in the scene
    for tpl_path in template_files:
        # Load template image and convert to grayscale
        tpl = cv2.imread(str(tpl_path))
        if tpl is None:
            print(f"Skipping unreadable template: {tpl_path}")
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        # Find all locations where the template matches above the threshold
        boxes, scores = match_template_once(scene_gray, tpl_gray, threshold=threshold)

        # Skip to next template if no matches found
        if not boxes:
            continue

        # Remove overlapping detections to get the best matches
        keep_idxs = m1.non_max_suppression(boxes, scores, iou_threshold=0.35)
        kept_boxes = [boxes[i] for i in keep_idxs]
        kept_scores = [scores[i] for i in keep_idxs]

        # Add all kept detections to the list with their template name as label
        label = tpl_path.stem
        for b, s in zip(kept_boxes, kept_scores):
            all_detections.append((b, s, label))

    # Stop if no detections were found across all templates
    if not all_detections:
        print(f"No objects detected (threshold={threshold}).")
        return

    # Draw green boxes and labels for all detected objects
    drawn = draw_detections(scene, all_detections)

    # Extract the bounding boxes from all detections
    boxes_to_blur = [det[0] for det in all_detections]

    # Save the image with detection boxes but no blur applied
    base_no_blur, ext_no_blur = os.path.splitext(str(scene_path))
    out_no_blur = f"{base_no_blur}_matched_all{ext_no_blur}"
    cv2.imwrite(out_no_blur, drawn)
    print(f"Detections-only image saved to: {out_no_blur}")

    # Apply Gaussian blur to the detected regions
    final = blur_regions(drawn, boxes_to_blur, blur_multiplier=blur_multiplier)

    # Save the final image with detection boxes and blurred regions
    base, ext = os.path.splitext(str(scene_path))
    out_path = f"{base}_matched_all_blurred{ext}"
    cv2.imwrite(out_path, final)
    print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
