"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 2 Assignment Part 1 - Object Detection with Template Matching

The purpose of this script is to detect an object using 
template matching based on correlation score. The program 
will find one or more instances of the template object 
within a scene(s) images. 

For this part of the Module 2, I have used OpenCV's cross 
correlation for comparing the template with the scene. 
Given a threshold value, any calculated correlation score 
above this value will be considered a match. Once a match 
has been found, a box surrounding the identified object in 
the scene will be displayed to the user and saved

Usage:
    1. Run the script - python module2_part1.py
    2. Provide the file path for the scene image with the object to be detected
	3. Provide the file path for the template image specifying the object
    4. Using threshold 0.8
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np


def non_max_suppression(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thresh: float = 0.4):
    """Simple non-max suppression for axis-aligned boxes.

    boxes - list of (x1, y1, x2, y2)
    scores - matching score for each box
    Returns: indices of boxes to keep
    """
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=float)
    scores_arr = np.array(scores, dtype=float)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def match_template(
    scene_path: str,
    template_path: str,
    threshold: float = 0.8,
    save_result: bool = True,
    *,
    draw_all: bool = True,
) -> bool:
    """Run normalized cross-correlation template matching.

    Returns True if at least one match >= threshold exists, otherwise False.
    Saves an annotated image if save_result=True.
    """
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    scene = cv2.imread(scene_path)
    template = cv2.imread(template_path)
    if scene is None:
        raise ValueError(f"Failed to load scene image: {scene_path}")
    if template is None:
        raise ValueError(f"Failed to load template image: {template_path}")

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = template_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]
    if th > sh or tw > sw:
        raise ValueError("Template is larger than scene — matching won't work.")

    # Use normalized cross-correlation
    res = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCORR_NORMED)

    # Locate all positions where value >= threshold
    # Get all locations where the response >= threshold
    y_idxs, x_idxs = np.where(res >= threshold)

    # If the image is large / threshold low this set may be huge; to keep
    # processing robust we only keep the top-K candidate positions by score.
    boxes = []
    scores = []
    for (y, x) in zip(y_idxs, x_idxs):
        boxes.append((int(x), int(y), int(x + tw), int(y + th)))
        scores.append(float(res[y, x]))

    # Keep only the top candidates (pre-NMS) to avoid extreme slowdowns
    max_candidates = 50
    if len(scores) > max_candidates:
        order = np.argsort(scores)[::-1][:max_candidates]
        boxes = [boxes[i] for i in order]
        scores = [scores[i] for i in order]

    found = len(boxes) > 0

    # If there are many overlapping boxes, cluster them via NMS
    if found:
        keep_idxs = non_max_suppression(boxes, scores, iou_thresh=0.35)
        kept_boxes = [boxes[i] for i in keep_idxs]
        kept_scores = [scores[i] for i in keep_idxs]

        # Decide whether to draw all kept boxes or only the highest-confidence one
        out = scene.copy()
        if not draw_all:
            # keep only the single highest-score detection (this forces the matcher
            # to scan the full response map and still return the most confident match)
            best_idx = int(np.argmax(kept_scores))
            kept_boxes = [kept_boxes[best_idx]]
            kept_scores = [kept_scores[best_idx]]

        # Find the highest-confidence kept detection (if any) so we can highlight it
        best_idx = 0
        if kept_scores:
            best_idx = int(np.argmax(kept_scores))

        # Draw all detections first (green). Then draw the best one with a thicker red box
        for i, (box, score) in enumerate(zip(kept_boxes, kept_scores)):
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            text = f"{score:.2f}"
            cv2.putText(out, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Highlight the single best detection in red with a thicker line and label
        if kept_boxes:
            bx = kept_boxes[best_idx]
            bscore = kept_scores[best_idx]
            x1, y1, x2, y2 = bx
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200), 3)
            label = f"BEST {bscore:.2f}"
            cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        if save_result:
            base, ext = os.path.splitext(scene_path)
            # Use template file name to avoid overwriting results when multiple
            # templates are matched against the same scene image.
            tpl_name = os.path.splitext(os.path.basename(template_path))[0]
            out_path = f"{base}_matched_{tpl_name}{ext}"
            cv2.imwrite(out_path, out)
            print(f"Result saved to: {out_path}")

        # Explain what we found
        if draw_all:
            print(f"Detected {len(kept_boxes)} match(es) (threshold={threshold}).")
        else:
            print(f"Detected 1 best match (score={kept_scores[0]:.2f}) (threshold={threshold}).")
        return True

    print(f"No matches found (threshold={threshold}).")
    return False


def main() -> None:
    # CLI usage: optionally accept arguments: scene_path [template_image|template_folder]
    # If none provided, fall back to interactive prompts (previous behaviour).
    # If the 2nd argument is a directory, all files beginning with 'template_' inside
    # that folder will be tried as templates.
    # Example:
    #   python module2_part1.py scene.jpg templates_folder/
    #   python module2_part1.py scene.jpg template_object.jpg

    # Prompt for both scene and template paths (user must enter them each time)
    print("Template-matching detection (correlation-based)")
    print()
    print("Provide the full path (or relative to the repo root) for both:")
    print(" - the scene image that contains the object to search for")
    print(" - the template image containing the object to match")
    print()

    # Require the user to enter both paths explicitly (no defaults).
    def prompt_for_path(prompt_text: str) -> str:
        while True:
            p = input(prompt_text).strip()
            if p and os.path.isfile(p):
                return p
            print("Path missing or file does not exist. Please enter a valid file path.")

    # Check for optional command line args
    import sys

    threshold = 0.8

    if len(sys.argv) >= 3:
        # Called as: python module2_part1.py <scene> <template_or_folder>
        scene_path = sys.argv[1]
        tpl_arg = sys.argv[2]

        if not os.path.isfile(scene_path):
            print(f"Scene file not found: {scene_path}")
            return

        # If the second arg is a directory: scan for files starting with 'template_'
        if os.path.isdir(tpl_arg):
            found_any = False
            matches_overall = False
            template_files = sorted(
                [os.path.join(tpl_arg, f) for f in os.listdir(tpl_arg) if f.startswith("template_")]
            )
            if not template_files:
                print(f"No template files found in folder '{tpl_arg}' (looking for files starting with 'template_').")
                return

            print(f"Found {len(template_files)} template file(s) in folder '{tpl_arg}'. Running matches...")
            for tfile in template_files:
                try:
                    matched = match_template(scene_path, tfile, threshold=threshold, save_result=True)
                    found_any = True
                    if matched:
                        matches_overall = True
                        print(f"Template {tfile} matched in scene.")
                    else:
                        print(f"Template {tfile} did NOT match.")
                except Exception as e:
                    print(f"Skipping template {tfile} — error: {e}")

            if not found_any:
                print("No candidate templates were processed.")
            elif matches_overall:
                print("At least one template matched in the scene.")
            else:
                print("No templates matched the scene.")
        else:
            # Treat as a single template image
            if not os.path.isfile(tpl_arg):
                print(f"Template not found: {tpl_arg}")
                return

            print("Doing the matching... please wait")
            try:
                matched = match_template(scene_path, tpl_arg, threshold=threshold, save_result=True)
                if matched:
                    print("Object detected in the scene.")
                else:
                    print("Object NOT detected in the scene.")
            except Exception as e:
                print(f"Error: {e}")

    else:
        # No CLI args: keep previous interactive behaviour
        def prompt_for_path(prompt_text: str) -> str:
            while True:
                p = input(prompt_text).strip()
                if p and os.path.isfile(p):
                    return p
                print("Path missing or file does not exist. Please enter a valid file path.")

        scene_path = prompt_for_path("Scene file path: ")
        template_path = prompt_for_path("Template file path: ")
        print("Doing the matching... please wait")

        try:
            matched = match_template(scene_path, template_path, threshold=threshold, save_result=True)
            if matched:
                print("Object detected in the scene.")
            else:
                print("Object NOT detected in the scene.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

