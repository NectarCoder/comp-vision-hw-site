"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 1 - Image Stitching

The purpose of this script is to perform image stitching 
on 8 portrait orientation images, assuming those 8 images 
are overlapping. OpenCV's SIFT feature detection has been 
used for identification of key points in the overlapping 
images. Image rotation, scaling and translation were used 
for preventing distortion in the final result. Mask erosion 
was also added additionally to remove any thin black lines 
that were present in the border of the individual images.

Usage:
    1. Images should be named in ascending order (left to right, 1 to 8) and placed together in a folder
    2. Run the script - python module4_part1.py
    3. Give the file path of the folder
    4. Script will process the images and stitch them together to create a panorama image
    5. The final result is stored as stitched_result_final.jpg and displayed to the user in a pop up window
"""

import cv2
import numpy as np
import glob
import os
import sys
import re

sys.setrecursionlimit(2000)

# --- CONFIGURATION ---
WORK_WIDTH = 800
# Higher values remove thicker lines but might eat into image details. 1 or 2 is usually enough.
SEAM_EROSION_ITERATIONS = 1 

def resize_image(image, width=None):
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
    # Using slightly higher contrast threshold for cleaner features on grass/trees
    sift = cv2.SIFT_create(contrastThreshold=0.03)
    kps, features = sift.detectAndCompute(gray, None)
    return kps, features

def match_keypoints_affine(kpsA, kpsB, featuresA, featuresB, ratio=0.75):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
    
    matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            matches.append(m)

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        
        # Use Affine Estimation (Rigid + Scale) for stability
        (M, inliers) = cv2.estimateAffinePartial2D(ptsA, ptsB)
        
        if M is None: return None

        # Convert to 3x3 for matrix multiplication
        H = np.vstack([M, [0, 0, 1]])
        status = np.ones(len(matches), dtype=np.uint8)
        return (matches, H, status)
    
    return None

def crop_black_borders(panorama):
    if panorama is None or panorama.size == 0: return None
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Apply a little dilation to the threshold to ensure we don't cut off valid dark pixels
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    coords = cv2.findNonZero(thresh)
    if coords is None: return panorama
    x, y, w, h = cv2.boundingRect(coords)
    return panorama[y:y+h, x:x+w]

def main():
    print("=== Assignment 4: Final Polished Stitching ===")
    
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
    
    original_images = [cv2.imread(p) for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_images = [img for img in original_images if img is not None]
    
    if len(original_images) < 2:
        print("[ERROR] Need at least 2 images.")
        return

    # Assuming standard Left->Right name order based on previous conversation
    print(f"Processing {len(original_images)} images...")
    images = [resize_image(img, width=WORK_WIDTH) for img in original_images]
    
    # --- 2. Compute Transforms ---
    homographies = [np.identity(3)]
    
    for i in range(len(images) - 1):
        print(f"Aligning Image {i+2} to {i+1}...")
        kps_prev, feats_prev = detect_and_describe(images[i])
        kps_curr, feats_curr = detect_and_describe(images[i+1])

        result = match_keypoints_affine(kps_curr, kps_prev, feats_curr, feats_prev)

        if result is None:
            print(f"[ERROR] Could not align Image {i+1} and {i+2}.")
            return
        
        (_, H_curr_to_prev, _) = result
        homographies.append(homographies[i].dot(H_curr_to_prev))

    # --- 3. Re-Center & Canvas ---
    print("Calculating optimal canvas...")
    mid_idx = len(images) // 2
    H_mid_to_0 = homographies[mid_idx]
    H_0_to_mid = np.linalg.inv(H_mid_to_0)

    new_homographies = []
    all_corners = []
    for i, H in enumerate(homographies):
        H_new = H_0_to_mid.dot(H)
        new_homographies.append(H_new)
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_new))

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    output_width, output_height = x_max - x_min, y_max - y_min
    
    print(f"Final Size: {output_width} x {output_height}")

    # --- 4. Stitching with Seam Fix ---
    print("Stitching and blending seams...")
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Kernel for eroding the edge of images to remove dark borders
    erosion_kernel = np.ones((3, 3), np.uint8)

    # Draw from extremities inwards to center
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - mid_idx), reverse=True)

    for i in draw_order:
        img = images[i]
        H_final = H_translation.dot(new_homographies[i])
        
        # Use INTER_LINEAR for smoother warping
        warped = cv2.warpPerspective(img, H_final, (output_width, output_height), flags=cv2.INTER_LINEAR)
        
        # Create mask and ERODE it to shave off dark edges
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # --- THE FIX IS HERE ---
        # Shrink the mask slightly so we don't copy the dark interpolation border
        mask = cv2.erode(mask, erosion_kernel, iterations=SEAM_EROSION_ITERATIONS)
        
        # Copy valid pixels based on eroded mask
        panorama[mask > 0] = warped[mask > 0]

    # --- 5. Finish ---
    print("Cropping...")
    panorama = crop_black_borders(panorama)
    
    output_path = os.path.join(img_dir, "stitched_result_final.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Saved: {output_path}")

    if panorama.shape[1] > 1800:
        panorama = resize_image(panorama, width=1800)
    
    cv2.imshow("Final Result", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()