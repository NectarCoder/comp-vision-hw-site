"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 1 - Image Stitching

The purpose of the script is to create a panorama by stitching
together several overlapping landscape orientation images. It
should be comparable to the panorama mode on a smartphone.

Usage:
    1. First make sure that the landscape images (at least 4) are in a folder
    2. Run the script - python module4_part1.py
    3. When prompted enter the path to the folder - path/to/folder/
    4. Images stitched left to right
    5. Final panorama will be saved as stitched_result.jpg and displayed
"""

import cv2
import numpy as np
import glob
import os
import sys
import re

sys.setrecursionlimit(2000)

# --- CONFIGURATION ---
# Resize images to this width for processing. 
# 800 is the "sweet spot" for speed vs quality in assignments.
WORK_WIDTH = 800 

def resize_image(image, width=None):
    """Resizes an image maintaining aspect ratio."""
    (h, w) = image.shape[:2]
    if width is None:
        return image
    r = width / float(w)
    dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(gray, None)
    return kps, features

def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reproj_thresh=4.0):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
    
    matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            matches.append(m)

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        
        if H is None: return None

        # --- STABILITY CHECKS ---
        # 1. Check Determinant (Zoom). Must be close to 1.0 (no massive scaling)
        det = np.linalg.det(H[:2, :2])
        if det < 0.5 or det > 2.0:
            print(f"   [Warning] Transformation implies >2x zoom (det={det:.2f}). Rejecting.")
            return None
            
        # 2. Check Translation. If shift > image width, it's a bad match.
        # H[0, 2] is x-translation.
        if abs(H[0, 2]) > 2000: # 2000 is generous for 800px width
             print(f"   [Warning] massive translation detected ({H[0,2]}). Rejecting.")
             return None

        return (matches, H, status)
    
    return None

def crop_panorama(panorama):
    if panorama is None or panorama.size == 0: return None
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None: return panorama
    x, y, w, h = cv2.boundingRect(coords)
    return panorama[y:y+h, x:x+w]

def main():
    print("=== Assignment 4: Stable Image Stitching ===")
    
    # --- 1. Load Images ---
    relative_img_dir = input("Enter path to image folder: ").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.normpath(os.path.join(script_dir, relative_img_dir))

    if not os.path.isdir(img_dir):
        print(f"[ERROR] Directory not found: {img_dir}")
        return

    print("Loading images...")
    image_paths = glob.glob(os.path.join(img_dir, "*"))
    image_paths.sort(key=natural_sort_key)
    
    # LOAD AND RESIZE IMMEDIATELY
    original_images = [cv2.imread(p) for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_images = [img for img in original_images if img is not None]
    
    if len(original_images) < 2:
        print("[ERROR] Need at least 2 images.")
        return

    print(f"Loaded {len(original_images)} images. Resizing to width {WORK_WIDTH}px for stability...")
    images = [resize_image(img, width=WORK_WIDTH) for img in original_images]
    
    # --- 2. Compute Chained Homographies ---
    homographies = [np.identity(3)]
    
    for i in range(len(images) - 1):
        print(f"Matching image {i+2} to {i+1}...")
        kps_prev, feats_prev = detect_and_describe(images[i])
        kps_curr, feats_curr = detect_and_describe(images[i+1])

        result = match_keypoints(kps_curr, kps_prev, feats_curr, feats_prev)

        if result is None:
            print(f"[ERROR] Failed to match image {i+1} and {i+2}.")
            return
        
        (matches, H_curr_to_prev, status) = result
        
        # Chain
        H_prev_to_0 = homographies[i]
        H_curr_to_0 = H_prev_to_0.dot(H_curr_to_prev)
        homographies.append(H_curr_to_0)

    print("Re-centering coordinate system...")

    # --- 3. Re-center ---
    mid_index = len(images) // 2
    H_mid_to_0 = homographies[mid_index]
    H_0_to_mid = np.linalg.inv(H_mid_to_0)

    new_homographies = []
    for H in homographies:
        new_homographies.append(H_0_to_mid.dot(H))
    homographies = new_homographies

    # --- 4. Calculate Canvas ---
    print("Calculating canvas size...")
    all_corners = []
    for i, H in enumerate(homographies):
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    output_width = x_max - x_min
    output_height = y_max - y_min

    print(f"Final Panorama Size: {output_width} x {output_height}")
    
    if output_width > 30000 or output_height > 30000:
        print("[ERROR] Size too large. The overlap between images might be too small.")
        return

    # --- 5. Warp ---
    print("Stitching...")
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    draw_order = sorted(range(len(images)), key=lambda i: abs(i - mid_index), reverse=True)

    for i in draw_order:
        img = images[i]
        H_final = H_translation.dot(homographies[i])
        cv2.warpPerspective(img, H_final, (output_width, output_height),
                            dst=panorama, borderMode=cv2.BORDER_TRANSPARENT)

    # --- 6. Finish ---
    panorama = crop_panorama(panorama)
    output_path = os.path.join(img_dir, "stitched_result.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Saved to {output_path}")

    cv2.imshow("Result", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()