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
"""

import cv2
import numpy as np
import glob
import os

def detect_and_describe(image):
    """Detects keypoints and computes descriptors using SIFT."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(gray, None)
    return kps, features

def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reproj_thresh=4.0):
    """
    Matches features between two sets of keypoints and computes the Homography.
    """
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)
    
    matches = []
    # Lowe's ratio test to filter for good matches
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            matches.append(m)

    # Homography calculation requires at least 4 matches
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        
        # Compute the homography matrix using RANSAC
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        
        return (matches, H, status)
    
    return None

def stitch(left_img, right_img):
    """
    Stitches the right image to the left image.
    """
    # Find keypoints and descriptors for both images
    kps_left, feats_left = detect_and_describe(left_img)
    kps_right, feats_right = detect_and_describe(right_img)
    
    # Match features and compute homography to map right_img to left_img
    result = match_keypoints(kps_right, kps_left, feats_right, feats_left)
    
    if result is None:
        print("[WARNING] Not enough matches found. Skipping pair.")
        return left_img
        
    (matches, H, status) = result
    
    # Get dimensions of both images
    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]
    
    # Create a new canvas to hold the stitched result
    # The canvas is wide enough to hold both images side-by-side
    panorama = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype="uint8")
    
    # Place the left image on the canvas
    panorama[0:h_left, 0:w_left] = left_img
    
    # Warp the right image onto the canvas, using the calculated homography.
    # The BORDER_TRANSPARENT flag ensures that the warped image overwrites the
    # existing canvas pixels only where the source image has content.
    cv2.warpPerspective(right_img, H, (panorama.shape[1], panorama.shape[0]), 
                        dst=panorama, borderMode=cv2.BORDER_TRANSPARENT)
    
    return panorama

def crop_panorama(panorama):
    """Crops the black border from the stitched panorama."""
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    
    # Find all non-black pixels
    coords = cv2.findNonZero(gray)
    if coords is None:
        # Return original image if it's all black
        return panorama

    # Get the bounding box for all non-black pixels
    x, y, w, h = cv2.boundingRect(coords)
    return panorama[y:y+h, x:x+w]

def main():
    print("=== Assignment 4: High-Quality Image Stitching ===")
    
    img_dir = input("Enter path to image folder: ").strip()
    if not os.path.isdir(img_dir):
        print(f"Directory not found: {img_dir}")
        return

    # Load images, assuming they are named in alphabetical, left-to-right order
    print("Loading images...")
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    images = [cv2.imread(p) for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [img for img in images if img is not None]
    
    if len(images) < 2:
        print("Error: Need at least 2 images to stitch.")
        return

    print(f"Loaded {len(images)} images. Starting stitching process...")
    
    # Incrementally stitch images from left to right
    panorama = images[0]
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)}...")
        panorama = stitch(panorama, images[i])

    print("Stitching complete. Cropping final panorama...")
    
    # Crop the final result to remove black borders
    panorama = crop_panorama(panorama)
    
    print("Process finished. Displaying result.")
    
    # Save and show the final result
    cv2.imwrite("stitched_result.jpg", panorama)
    cv2.imshow("Resulting Panorama", panorama)
    
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()