"""
CSC 8830 Assignment 4 - Problem 2: SIFT From Scratch & RANSAC
Author: [Your Name]

Description:
    1. Implements a simplified version of SIFT feature extraction from scratch:
       - Scale Space Construction (Gaussian Pyramid)
       - Difference of Gaussians (DoG)
       - Keypoint Extrema Detection
    2. Implements a custom RANSAC algorithm for Homography estimation.
    3. Compares results with OpenCV's optimized SIFT implementation.

Usage:
    python assignment4_sift_scratch.py
"""

import cv2
import numpy as np
import random

class SIFT_Scratch:
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6):
        self.num_octaves = num_octaves
        self.num_scales = num_scales # Scales per octave
        self.sigma = sigma

    def generate_gaussian_pyramid(self, image):
        """ Generates Gaussian Pyramid (Scale Space) """
        pyramid = []
        temp_img = image.copy()

        for _ in range(self.num_octaves):
            octave = []
            octave.append(temp_img)
            
            # Generate blurred versions for this octave
            for i in range(1, self.num_scales):
                # Scale sigma
                k = 2 ** (i / (self.num_scales - 3)) 
                curr_sigma = k * self.sigma
                blurred = cv2.GaussianBlur(temp_img, (0, 0), curr_sigma, curr_sigma)
                octave.append(blurred)
            
            pyramid.append(octave)
            
            # Downsample for next octave
            temp_img = cv2.resize(temp_img, (int(temp_img.shape[1]/2), int(temp_img.shape[0]/2)))
        
        return pyramid

    def generate_dog_pyramid(self, gaussian_pyramid):
        """ Generates Difference of Gaussian (DoG) Pyramid """
        dog_pyramid = []
        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(len(octave) - 1):
                # DoG is simply subtraction of adjacent scales
                dog = cv2.subtract(octave[i+1], octave[i])
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
        return dog_pyramid

    def find_extrema(self, dog_pyramid):
        """ Find local extrema (maxima/minima) in DoG space """
        keypoints = []
        
        # Iterate over octaves
        for octave_idx, dog_octave in enumerate(dog_pyramid):
            # Need 3 images to compare (prev, curr, next)
            for i in range(1, len(dog_octave) - 1):
                prev_img = dog_octave[i-1]
                curr_img = dog_octave[i]
                next_img = dog_octave[i+1]
                
                # Iterate pixels (ignoring borders)
                rows, cols = curr_img.shape
                for r in range(1, rows - 1):
                    for c in range(1, cols - 1):
                        val = curr_img[r, c]
                        
                        # Get 26 neighbors (9 in prev, 8 in curr, 9 in next)
                        neighbors = []
                        neighbors.extend(prev_img[r-1:r+2, c-1:c+2].flatten())
                        neighbors.extend(curr_img[r-1:r+2, c-1:c+2].flatten())
                        neighbors.extend(next_img[r-1:r+2, c-1:c+2].flatten())
                        
                        # Remove self from comparison
                        # (The center pixel is in the middle of the curr_img block)
                        # Simplification: just check if val is max or min of block
                        
                        is_max = val >= np.max(neighbors)
                        is_min = val <= np.min(neighbors)
                        
                        if is_max or is_min:
                            # Store keypoint: (x, y, size)
                            # Map back to original image size based on octave
                            scale_factor = 2 ** octave_idx
                            orig_x = c * scale_factor
                            orig_y = r * scale_factor
                            keypoints.append(cv2.KeyPoint(float(orig_x), float(orig_y), 10 * scale_factor))
                            
        return keypoints

# --- Custom RANSAC Implementation ---

def compute_homography_custom(src_pts, dst_pts):
    """ Wrapper for finding homography (using DLT algorithm internally usually) """
    # For simplicity in 'scratch' implementation, we use findHomography with 0 (no robust method)
    # to just get the matrix from 4 points, then we do RANSAC manually below.
    H, _ = cv2.findHomography(src_pts, dst_pts, 0)
    return H

def ransac_optimization(src_pts, dst_pts, threshold=5.0, max_iterations=1000):
    """ 
    Implementation of RANSAC for Homography 
    """
    best_H = None
    max_inliers = 0
    
    num_points = len(src_pts)
    if num_points < 4:
        return None, 0

    src_pts_h = np.hstack((src_pts, np.ones((num_points, 1)))) # Homogeneous coords

    for _ in range(max_iterations):
        # 1. Randomly select 4 points
        indices = random.sample(range(num_points), 4)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        # 2. Compute Homography for this sample
        H = compute_homography_custom(src_sample, dst_sample)
        if H is None: continue
        
        # 3. Project all points using H
        projected_pts = src_pts_h @ H.T
        # Normalize (divide by z)
        projected_pts = projected_pts / (projected_pts[:, 2:3] + 1e-7)
        projected_pts = projected_pts[:, :2]
        
        # 4. Calculate errors (Euclidean distance)
        errors = np.linalg.norm(dst_pts - projected_pts, axis=1)
        
        # 5. Count inliers
        current_inliers = np.sum(errors < threshold)
        
        if current_inliers > max_inliers:
            max_inliers = current_inliers
            best_H = H
            
    return best_H, max_inliers


def main():
    print("=== Assignment 4: SIFT & RANSAC from Scratch ===")
    
    img_path = input("Enter path to image (e.g., box.png): ").strip()
    if not os.path.exists(img_path):
        print("Image not found.")
        return
        
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vis_scratch = img.copy()
    img_vis_opencv = img.copy()

    # --- Part 1: SIFT From Scratch ---
    print("\n[1] Running Custom SIFT (Scale Space & Extrema)...")
    print("    (This may take a moment due to pure Python loops)")
    
    my_sift = SIFT_Scratch(num_octaves=3, num_scales=4)
    g_pyr = my_sift.generate_gaussian_pyramid(gray)
    dog_pyr = my_sift.generate_dog_pyramid(g_pyr)
    keypoints_custom = my_sift.find_extrema(dog_pyr)
    
    print(f"    Custom SIFT detected {len(keypoints_custom)} keypoints.")
    cv2.drawKeypoints(img_vis_scratch, keypoints_custom, img_vis_scratch, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # --- Part 2: OpenCV SIFT ---
    print("\n[2] Running OpenCV SIFT...")
    sift_cv = cv2.SIFT_create()
    keypoints_cv, descriptors_cv = sift_cv.detectAndCompute(gray, None)
    
    print(f"    OpenCV SIFT detected {len(keypoints_cv)} keypoints.")
    cv2.drawKeypoints(img_vis_opencv, keypoints_cv, img_vis_opencv, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # --- Part 3: RANSAC Comparison ---
    print("\n[3] Testing RANSAC (Self-Match Test)...")
    # We simulate a transformation to test RANSAC
    rows, cols = gray.shape
    # Create a synthetic transformation (rotation + translation)
    M_synth = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    warped_gray = cv2.warpAffine(gray, M_synth, (cols, rows))
    
    # Detect features in warped image
    kp_w, desc_w = sift_cv.detectAndCompute(warped_gray, None)
    
    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_cv, desc_w, k=2)
    
    good_matches = []
    pts_src = []
    pts_dst = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts_src.append(keypoints_cv[m.queryIdx].pt)
            pts_dst.append(kp_w[m.trainIdx].pt)
            
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)
    
    if len(pts_src) > 4:
        # Run Custom RANSAC
        H_custom, inliers = ransac_optimization(pts_src, pts_dst)
        print(f"    Custom RANSAC Inliers: {inliers}/{len(pts_src)}")
        print("    Custom Homography:\n", H_custom)
    else:
        print("    Not enough matches for RANSAC test.")

    # --- Visualization ---
    cv2.imshow("Custom SIFT Extrema", img_vis_scratch)
    cv2.imshow("OpenCV SIFT Extrema", img_vis_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()