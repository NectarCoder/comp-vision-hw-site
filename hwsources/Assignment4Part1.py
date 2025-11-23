"""
CSC 8830 Assignment 4 - Problem 1: Image Stitching
Author: [Your Name]

Description:
    This script implements image stitching to create a panorama from a set of 
    overlapping images. It performs the following steps:
    1. Detects keypoints and descriptors (using SIFT).
    2. Matches features between adjacent images.
    3. Computes the Homography matrix using RANSAC.
    4. Warps one image to align with the other and blends them.
    
    It is designed to stitch a sequence of images (e.g., 'img1.jpg', 'img2.jpg'...)
    taken in a horizontal sweep.

Usage:
    1. Place your 4+ images in a folder.
    2. Run: python assignment4_stitching.py
    3. Provide the folder path when prompted.
"""

import cv2
import numpy as np
import glob
import os
import sys

def detect_and_describe(image):
    """
    Helper: Detects keypoints and computes descriptors using SIFT.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(gray, None)
    return kps, features

def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reproj_thresh=4.0):
    """
    Matches features between two images and computes Homography.
    """
    # BruteForce Matcher
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
    
    matches = []
    # Lowe's ratio test
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            matches.append(m)

    # Need at least 4 matches to compute homography
    if len(matches) > 4:
        # Construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        
        # Compute Homography (H) that maps ptsA to ptsB
        # RANSAC is used here to be robust against outliers
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        
        return (matches, H, status)
    
    return None

def stitch_pair(imgA, imgB):
    """
    Stitches two images together. 
    imgA: The image to be warped (right)
    imgB: The base image (left)
    """
    kpsA, featsA = detect_and_describe(imgA)
    kpsB, featsB = detect_and_describe(imgB)
    
    result = match_keypoints(kpsA, kpsB, featsA, featsB)
    
    if result is None:
        print("[WARNING] Not enough matches found between pair.")
        return imgB
        
    (matches, H, status) = result
    
    # Get dimensions
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    
    # Warp imgA to imgB's coordinate space
    # The size of the new canvas needs to be large enough
    # For a simple horizontal stitch, we add widths
    panorama = cv2.warpPerspective(imgA, H, (wA + wB, hA))
    
    # Place imgB on the panorama
    panorama[0:hB, 0:wB] = imgB
    
    # Optional: Simple blending could be added here to remove seam lines
    # For this assignment, direct overlay is usually sufficient to demonstrate logic
    
    # Crop black regions (simple approach)
    # Find all non-black points
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.CV_RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        panorama = panorama[y:y+h, x:x+w]
        
    return panorama

def main():
    print("=== Assignment 4: Image Stitching ===")
    
    img_dir = input("Enter path to image folder: ").strip()
    if not os.path.exists(img_dir):
        print("Directory not found.")
        return

    # Load images
    # Assuming images are named sequentially or ordered alphabetically
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    images = []
    
    for path in image_paths:
        # Basic check for image extensions
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(path)
            if img is not None:
                # Resize to speed up processing if images are huge
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
                images.append(img)
    
    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return

    print(f"Loaded {len(images)} images. Stitching...")
    
    # Incremental stitching: stitch 1 and 2, result with 3, etc.
    # Note: Order matters! This assumes left-to-right sequence.
    # If warping looks wrong, try reversing the list: images = images[::-1]
    
    panorama = images[0]
    
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}...")
        # Stitch current panorama with next image
        # We treat 'panorama' as the 'right' image being warped in this simplistic loop 
        # or vice versa depending on match direction. 
        # Standard approach: Align Img2 to Img1.
        panorama = stitch_pair(images[i], panorama)

    print("Stitching Complete.")
    cv2.imshow("Resulting Panorama", panorama)
    cv2.imwrite("stitched_result.jpg", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()