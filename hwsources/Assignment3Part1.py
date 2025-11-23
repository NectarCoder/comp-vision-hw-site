"""
CSC 8830 Assignment 3 - Problems 1, 2, & 3
Author: [Your Name]

Description:
    A comprehensive toolkit to process a dataset of images for:
    1. Gradients (Magnitude, Angle) & Laplacian of Gaussian (LoG).
    2. Feature Detection (Canny Edges, Harris Corners).
    3. Exact Boundary Detection (Contours).

Usage:
    python assignment3_cv_toolkit.py
    (Follow prompts to provide image directory)
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def process_gradients_log(img_gray):
    """ Problem 1: Gradient Mag/Angle and LoG """
    # 1. Gradients (Sobel)
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and Angle
    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    
    # Normalize Magnitude for visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Laplacian of Gaussian (LoG)
    # Blur first (Gaussian), then Laplacian
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude_norm, angle, log_norm

def detect_features(img_gray):
    """ Problem 2: Edges and Corners """
    # 1. Edge Detection (Canny)
    edges = cv2.Canny(img_gray, 100, 200)
    
    # 2. Corner Detection (Harris)
    # dst returns a response map
    dst = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    # Dilate for marking corners clearly
    dst_dilated = cv2.dilate(dst, None)
    
    return edges, dst_dilated

def find_exact_boundaries(img_rgb, img_gray):
    """ Problem 3: Exact Object Boundaries (Contours) """
    # Preprocessing for better contour detection
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Otsu's thresholding automatically finds best threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find Contours
    contours, hierarchy = cv2.findContours(thresh, cv2.CV_RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy
    boundary_img = img_rgb.copy()
    cv2.drawContours(boundary_img, contours, -1, (0, 255, 0), 2)
    
    return boundary_img, contours

def main():
    print("=== Assignment 3: CV Toolkit ===")
    img_dir = input("Enter path to dataset folder (containing 10 images): ").strip()
    
    if not os.path.exists(img_dir):
        print(f"[ERROR] Directory '{img_dir}' not found.")
        return

    # Load all images
    extensions = ['*.jpg', '*.png', '*.jpeg']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    print(f"[INFO] Found {len(img_files)} images.")
    
    for fpath in img_files:
        filename = os.path.basename(fpath)
        print(f"\nProcessing {filename}...")
        
        img = cv2.imread(fpath)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- Q1: Gradients & LoG ---
        mag, ang, log = process_gradients_log(img_gray)
        
        # --- Q2: Features ---
        edges, corners = detect_features(img_gray)
        # Create visual for corners
        corners_vis = img_rgb.copy()
        corners_vis[corners > 0.01 * corners.max()] = [255, 0, 0]
        
        # --- Q3: Boundaries ---
        boundary_img, _ = find_exact_boundaries(img_rgb, img_gray)
        
        # --- Visualization (Matplotlib) ---
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Analysis: {filename}")
        
        # Row 1: Gradients
        plt.subplot(2, 4, 1); plt.imshow(img_rgb); plt.title("Original")
        plt.axis('off')
        plt.subplot(2, 4, 2); plt.imshow(mag, cmap='jet'); plt.title("Gradient Mag")
        plt.axis('off')
        plt.subplot(2, 4, 3); plt.imshow(ang, cmap='hsv'); plt.title("Gradient Angle")
        plt.axis('off')
        plt.subplot(2, 4, 4); plt.imshow(log, cmap='gray'); plt.title("LoG")
        plt.axis('off')
        
        # Row 2: Features & Boundaries
        plt.subplot(2, 4, 5); plt.imshow(edges, cmap='gray'); plt.title("Edges (Canny)")
        plt.axis('off')
        plt.subplot(2, 4, 6); plt.imshow(corners_vis); plt.title("Corners (Harris)")
        plt.axis('off')
        plt.subplot(2, 4, 7); plt.imshow(boundary_img); plt.title("Exact Boundary")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Pause for user
        # input("Press Enter for next image...")

if __name__ == "__main__":
    main()