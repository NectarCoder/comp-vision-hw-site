"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 1 - Object Detection through Correlation-Based Template Matching

The purpose of this script is to detect a given 
object (template) within an image (scene). For this 
part of the assignment, OpenCV's template matching 
functionality has been used. There were issues with 
identification due to the lighting differences between 
template and scene. As such I had to use OpenCV's normalized 
cross correlation mapping for getting accurate correlation scores.

Usage:
	1. Run the script - python module2_part1.py
    2. Enter the file path of the scene image
    3. Enter the file path of template image
    4. Script will save an output image with a 
       box drawn around the detected object if found

Notes:
	For better results make sure template image is close 
    in terms of scale and well as rotation/orientation 
    to the object in the scene.

Dependencies include OpenCV (cv2) and NumPy
"""

from __future__ import annotations
import os
from typing import List, Tuple
import cv2
import numpy as np

# cv2 and numpy let us open images and do the math for matching

def non_max_suppression(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thresh: float = 0.4):
    # return an empty list when nothing to suppress
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
    # keep the highest score in each group and skip overlapping ones
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
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # load the images now that paths are validated
    scene = cv2.imread(scene_path)
    template = cv2.imread(template_path)
    if scene is None:
        raise ValueError(f"Failed to load scene image: {scene_path}")
    if template is None:
        raise ValueError(f"Failed to load template image: {template_path}")

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # convert to gray so we only compare brightness data

    th, tw = template_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]
    if th > sh or tw > sw:
        raise ValueError("Template is larger than scene — matching won't work.")

    res = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCORR_NORMED)
    # res tells us how strong the template match is at each spot

    y_idxs, x_idxs = np.where(res >= threshold)
    # keep every location that passed the threshold so far

    boxes = []
    scores = []
    for (y, x) in zip(y_idxs, x_idxs):
        boxes.append((int(x), int(y), int(x + tw), int(y + th)))
        scores.append(float(res[y, x]))

    # keep the strongest few matches so we stay quick
    max_candidates = 50
    if len(scores) > max_candidates:
        order = np.argsort(scores)[::-1][:max_candidates]
        boxes = [boxes[i] for i in order]
        scores = [scores[i] for i in order]

    # remember if we found any candidates before suppressing overlaps
    found = len(boxes) > 0

    if found:
        # drop overlapping detections so we only draw one per object
        keep_idxs = non_max_suppression(boxes, scores, iou_thresh=0.35)
        kept_boxes = [boxes[i] for i in keep_idxs]
        kept_scores = [scores[i] for i in keep_idxs]

        # draw rectangles on a copy so we never mutate the raw scene image
        out = scene.copy()
        if not draw_all:
            best_idx = int(np.argmax(kept_scores))
            kept_boxes = [kept_boxes[best_idx]]
            kept_scores = [kept_scores[best_idx]]

        # remember which kept box has the highest score for the highlight
        best_idx = 0
        if kept_scores:
            best_idx = int(np.argmax(kept_scores))

        # draw every kept box with its score so user sees candidates
        for i, (box, score) in enumerate(zip(kept_boxes, kept_scores)):
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            text = f"{score:.2f}"
            cv2.putText(out, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if kept_boxes:
            # emphasize the single best match for quick pruning
            bx = kept_boxes[best_idx]
            bscore = kept_scores[best_idx]
            x1, y1, x2, y2 = bx
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200), 3)
            label = f"BEST {bscore:.2f}"
            cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        if save_result:
            base, ext = os.path.splitext(scene_path)
            tpl_name = os.path.splitext(os.path.basename(template_path))[0]
            out_path = f"{base}_matched_{tpl_name}{ext}"
            cv2.imwrite(out_path, out)
            print(f"Result saved to: {out_path}")

        if draw_all:
            print(f"Detected {len(kept_boxes)} match(es) (threshold={threshold}).")
        else:
            print(f"Detected 1 best match (score={kept_scores[0]:.2f}) (threshold={threshold}).")
        return True

    print(f"No matches found (threshold={threshold}).")
    return False


def main() -> None:
    print("Template-matching detection (correlation-based)")
    print()
    print("Provide the full path (or relative to the repo root) for both:")
    print(" - the scene image that contains the object to search for")
    print(" - the template image containing the object to match")
    print()
    # remind the user that both file paths are needed

    # helper so we can reuse the same prompt logic twice
    def prompt_for_path(prompt_text: str) -> str:
        while True:
            p = input(prompt_text).strip()
            if p and os.path.isfile(p):
                return p
            print("Path missing or file does not exist. Please enter a valid file path.")

    # allow the script to read CLI arguments if provided
    import sys

    threshold = 0.7

    # allow optional scene + template arguments for scripting
    if len(sys.argv) >= 3:
        scene_path = sys.argv[1]
        tpl_arg = sys.argv[2]

        if not os.path.isfile(scene_path):
            print(f"Scene file not found: {scene_path}")
            return

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
                    # try each template and remember if any matched
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
        # fallback to asking the user once we have no CLI values
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
            # run the matcher and report success or failure
            matched = match_template(scene_path, template_path, threshold=threshold, save_result=True)
            if matched:
                print("Object detected in the scene.")
            else:
                print("Object NOT detected in the scene.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

