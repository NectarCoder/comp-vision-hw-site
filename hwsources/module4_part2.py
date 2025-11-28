"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 2 - Image Stitching with SIFT and RANSAC (from scratch)

The purpose of the script is to perform image 
stitching on a sequence of overlapping portrait 
orientation images, the difference being SIFT and 
RANSAC were implemented from scratch

Some key details for the implementation :-
 - SIFT based on difference of Gaussian
 - RANSAC I used affine transformation based approach instead of homography
 - There was bowtie distortion, so I have applied centering based on whichever 
   image is in the middle of the sequence
 - Comparison is performed between manual SIFT/RANSAC implementation, as well as 
   OpenCV's SIFT/RANSAC methods

Usage:
    1. Images should be named in ascending order (left to right, 1 to 8) and placed in a folder
    2. Run the script - python module4_part2.py
    3. Enter the file path of the image folder
    4. The script will run the images through OpenCV's SIFT/RANSAC as well as the custom SIFT/RANSAC
    5. Final results (Custom Stitching vs. OpenCV Stitching) are displayed in a Matplotlib window
"""

import cv2
import numpy as np
import glob
import os
import re

# --- CONFIGURATION ---
WORK_WIDTH = 400 

# ==========================================
# PART A: SCRATCH SIFT CLASS
# ==========================================

class ScratchSIFT:
    def __init__(self, sigma=1.6, num_intervals=3, contrast_threshold=0.04, edge_threshold=10):
        self.sigma = sigma
        self.intervals = intervals
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

    # Get the Gaussian images - each image is progressively blurred
    def generate_gaussian_images(self, image, sigma):
        gaussian_images = [image] # Storing the image first
        k = 2 ** (1 / self.intervals) # Blurring scale factor
        # Generating self.intervals + 3 images so that there are enough extra layers on top/bottom for keypoint detection
        for i in range(self.intervals + 2):
            current_sigma = (k**i) * sigma #Calculation of sigma for current iteration
            blur = cv2.GaussianBlur(image, (0, 0), current_sigma) # Applying gaussian blur
            gaussian_images.append(blur)
        return gaussian_images

    def generate_dog(self, gaussian_images):
        """Generates Difference of Gaussian (DoG) to approximate Laplacian."""
        dog_images = []
        for i in range(len(gaussian_images) - 1):
            # Subtract adjacent scales to find features
            dog_images.append(gaussian_images[i+1] - gaussian_images[i])
        return dog_images

    def is_edge_like(self, image, r, c):
        """Rejects weak edge-like features using Hessian Matrix (Eigenvalue check)."""
        val = image[r, c]
        # Calculate gradients (Dxx, Dyy, Dxy)
        Dxx = image[r, c+1] + image[r, c-1] - 2 * val
        Dyy = image[r+1, c] + image[r-1, c] - 2 * val
        Dxy = (image[r+1, c+1] + image[r-1, c-1] - image[r-1, c+1] - image[r+1, c-1]) / 4.0
        
        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy**2
        
        # If curvature is high in one direction but low in other, it's an edge (bad feature)
        if det <= 0: return True
        score = (trace**2) / det
        thresh = ((self.edge_threshold + 1)**2) / self.edge_threshold
        return score >= thresh

    def find_keypoints(self, dog_images):
        """Detects extrema (maxima/minima) in 3D DoG scale space."""
        keypoints = []
        h, w = dog_images[0].shape
        # Iterate through scales (excluding top/bottom)
        for i in range(1, len(dog_images) - 1):
            prev, curr, nxt = dog_images[i-1], dog_images[i], dog_images[i+1]
            # Iterate pixels (ignoring border)
            for r in range(15, h - 15):
                for c in range(15, w - 15):
                    val = curr[r, c]
                    # Filter low contrast points
                    if abs(val) < self.contrast_threshold: continue
                    
                    # Check 26 neighbors (9 in prev, 8 in curr, 9 in nxt)
                    block = np.array([prev[r-1:r+2, c-1:c+2], curr[r-1:r+2, c-1:c+2], nxt[r-1:r+2, c-1:c+2]])
                    if val == block.max() or val == block.min():
                        # If extremum, check if it is just an edge
                        if not self.is_edge_like(curr, r, c):
                            keypoints.append(cv2.KeyPoint(float(c), float(r), size=10))
        return keypoints

    def get_interpolated_gradient(self, r, c, dx_img, dy_img):
        """Bilinear Interpolation to get sub-pixel gradient values."""
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        if r0 < 0 or c0 < 0 or r1 >= dx_img.shape[0] or c1 >= dx_img.shape[1]: return 0.0, 0.0
        
        # Weighted average of 4 nearest neighbors
        wr, wc = r - r0, c - c0
        dx = (1-wr)*(1-wc)*dx_img[r0, c0] + (1-wr)*wc*dx_img[r0, c1] + wr*(1-wc)*dx_img[r1, c0] + wr*wc*dx_img[r1, c1]
        dy = (1-wr)*(1-wc)*dy_img[r0, c0] + (1-wr)*wc*dy_img[r0, c1] + wr*(1-wc)*dy_img[r1, c0] + wr*wc*dy_img[r1, c1]
        return dx, dy

    def compute_descriptor(self, image, keypoints):
        """Generates 128-d rotation-invariant descriptors for keypoints."""
        descriptors = []
        image = image.astype(np.float32)
        
        # Pre-compute gradients
        dx_img = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        dy_img = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        mag_img, angle_img = cv2.cartToPolar(dx_img, dy_img, angleInDegrees=True)
        
        # 1. Assign Dominant Orientation (to make feature rotation invariant)
        final_kps = []
        for kp in keypoints:
            c, r = int(kp.pt[0]), int(kp.pt[1])
            hist = np.zeros(36) # 36 bins (10 degrees each)
            # Accumulate gradient magnitudes in local window
            for i in range(-4, 5):
                for j in range(-4, 5):
                    if 0 <= r+i < image.shape[0] and 0 <= c+j < image.shape[1]:
                        bin_idx = int(angle_img[r+i, c+j] / 10) % 36
                        hist[bin_idx] += mag_img[r+i, c+j]
            kp.angle = float(np.argmax(hist) * 10)
            final_kps.append(kp)

        # 2. Build 128-d Descriptor (4x4 grid, 8 orientation bins)
        # Gaussian weighting to give less importance to pixels far from center
        gauss_weight = cv2.getGaussianKernel(16, 8) @ cv2.getGaussianKernel(16, 8).T
        valid_kps = []
        
        for kp in final_kps:
            kp_c, kp_r, kp_angle = kp.pt[0], kp.pt[1], kp.angle
            angle_rad = np.deg2rad(-kp_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            desc_vec = np.zeros((4, 4, 8))
            valid_window = True
            
            # Rotate window to align with dominant orientation
            for i in range(16):
                for j in range(16):
                    r_grid, c_grid = i - 7.5, j - 7.5
                    c_rot = c_grid * cos_a - r_grid * sin_a
                    r_rot = c_grid * sin_a + r_grid * cos_a
                    sample_r, sample_c = kp_r + r_rot, kp_c + c_rot
                    
                    val_dx, val_dy = self.get_interpolated_gradient(sample_r, sample_c, dx_img, dy_img)
                    # Check boundary conditions
                    if val_dx == 0 and val_dy == 0 and (sample_r < 2 or sample_r > image.shape[0]-2):
                        valid_window = False; break
                        
                    val_mag = np.sqrt(val_dx**2 + val_dy**2)
                    val_ang = np.degrees(np.arctan2(val_dy, val_dx)) % 360
                    rel_ang = (val_ang - kp_angle) % 360
                    weight = val_mag * gauss_weight[i, j]
                    
                    # Trilinear interpolation (Soft Binning) to avoid boundary artifacts
                    r_bin, c_bin, o_bin = (i+0.5)/4.0 - 0.5, (j+0.5)/4.0 - 0.5, rel_ang/45.0
                    r0, c0, o0 = int(np.floor(r_bin)), int(np.floor(c_bin)), int(np.floor(o_bin))
                    dr, dc, do = r_bin - r0, c_bin - c0, o_bin - o0
                    
                    for r_idx, w_r in [(r0, 1-dr), (r0+1, dr)]:
                        if 0 <= r_idx < 4:
                            for c_idx, w_c in [(c0, 1-dc), (c0+1, dc)]:
                                if 0 <= c_idx < 4:
                                    for o_idx, w_o in [(o0, 1-do), ((o0+1), do)]:
                                        desc_vec[r_idx, c_idx, o_idx % 8] += weight * w_r * w_c * w_o
                if not valid_window: break
            
            if valid_window:
                # Normalize and Threshold to handle illumination changes
                vec = desc_vec.flatten()
                norm = np.linalg.norm(vec)
                if norm > 1e-6: vec /= norm
                vec[vec > 0.2] = 0.2 # Cap large gradients
                norm = np.linalg.norm(vec)
                if norm > 1e-6: vec /= norm
                descriptors.append(vec)
                valid_kps.append(kp)
                
        return valid_kps, np.array(descriptors, dtype=np.float32)

# ==========================================
# PART B: SCRATCH AFFINE RANSAC
# ==========================================

def scratch_ransac_affine(pts_src, pts_dst, threshold=5.0, max_iters=2000):
    """
    Custom RANSAC implementation using AFFINE transforms (3 points)
    instead of Homography (4 points). This prevents extreme stretching
    in panoramic stitching.
    """
    best_H = None
    max_inliers = 0
    best_mask = None
    n = pts_src.shape[0]
    
    # We need at least 3 points for Affine
    if n < 3: return None, None
    src, dst = np.squeeze(pts_src), np.squeeze(pts_dst)

    for _ in range(max_iters):
        # 1. Random Sample (3 points for Affine)
        idx = np.random.choice(n, 3, replace=False)
        p1, p2 = src[idx], dst[idx]
        
        # 2. Compute Affine Transform for these 3 points
        M = cv2.getAffineTransform(p1, p2)
        
        # Convert 2x3 Affine to 3x3 Homography format for matrix multiplication later
        H = np.vstack([M, [0, 0, 1]])

        # 3. Project all points using this model
        src_h = np.hstack((src, np.ones((n, 1))))
        pred_h = (H @ src_h.T).T
        pred = pred_h[:, :2] 
        
        # 4. Count Inliers (Points where dist < threshold)
        inliers = np.linalg.norm(dst - pred, axis=1) < threshold
        count = np.sum(inliers)
        
        # 5. Keep Best Model
        if count > max_inliers:
            max_inliers = count
            best_H = H
            best_mask = inliers
            
    return best_H, best_mask

def resize_image(image, width=None):
    """Resizes image maintaining aspect ratio."""
    (h, w) = image.shape[:2]
    if width is None: return image
    r = width / float(w)
    dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def natural_sort_key(s):
    """Sorts strings naturally (1, 2, 10) instead of ASCII (1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def detect_and_describe_custom(image, name="Image"):
    """Wrapper to call the custom ScratchSIFT logic."""
    print(f"   [Scratch] Detecting features in {name}...", end=" ", flush=True)
    sift = CustomSIFT()
    gray = image
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Generate features
    dog = sift.generate_dog(sift.generate_gaussian_images(gray.astype(np.float32), 1.6))
    kps_raw = sift.find_keypoints(dog)
    kps, features = sift.compute_descriptor(gray, kps_raw)
    print(f"Found {len(kps)} keypoints.")
    return kps, features

def match_keypoints_custom(kpsA, kpsB, featuresA, featuresB, ratio=0.8):
    """Matches descriptors using Euclidean distance and Lowe's Ratio Test."""
    if len(kpsA) == 0 or len(kpsB) == 0: return None
    
    matches = []
    # Brute-force Euclidean Distance matching
    for i in range(len(featuresA)):
        diff = featuresB - featuresA[i]
        dist = np.linalg.norm(diff, axis=1)
        idx_sorted = np.argsort(dist)
        
        # Lowe's Ratio test: Best match must be significantly better than 2nd best
        if dist[idx_sorted[0]] < ratio * dist[idx_sorted[1]]:
            matches.append(cv2.DMatch(i, idx_sorted[0], dist[idx_sorted[0]]))

    if len(matches) > 4:
        # Extract point coordinates
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        
        # Use Custom Scratch RANSAC (Affine) to filter outliers
        H, mask = scratch_ransac_affine(ptsA, ptsB)
        
        if H is None: return None
        return (matches, H, mask)
    
    return None

def crop_black_borders(panorama):
    """Trims excess black pixels from the stitched result."""
    if panorama is None or panorama.size == 0: return None
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    coords = cv2.findNonZero(thresh)
    if coords is None: return panorama
    x, y, w, h = cv2.boundingRect(coords)
    return panorama[y:y+h, x:x+w]

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def run_stitching_custom(images):
    print(f"\n--- Starting Custom Stitching on {len(images)} Images ---")
    
    # 1. Compute pairwise homographies (chaining from 1st image)
    homographies = [np.identity(3)]
    
    for i in range(len(images) - 1):
        print(f"\n[Step {i+1}/{len(images)-1}] Aligning Image {i+2} to {i+1}...")
        
        # Detect features using Custom SIFT
        kps_prev, feats_prev = detect_and_describe_custom(images[i], f"Img {i+1}")
        kps_curr, feats_curr = detect_and_describe_custom(images[i+1], f"Img {i+2}")
        
        # Match using Custom RANSAC
        result = match_keypoints_custom(kps_curr, kps_prev, feats_curr, feats_prev)
        
        if result is None:
            print(f"[ERROR] Could not align Image {i+1} and {i+2}. Not enough matches.")
            return None
            
        (_, H_curr_to_prev, _) = result
        print(f"   [Scratch] Alignment successful.")
        
        # Accumulate transforms relative to first image
        homographies.append(homographies[i].dot(H_curr_to_prev))

    # 2. Re-center canvas to prevent 'bowtie' distortion
    print("\n[Scratch] Calculating optimal canvas...")
    mid_idx = len(images) // 2
    H_mid_to_0 = homographies[mid_idx]
    H_0_to_mid = np.linalg.inv(H_mid_to_0)
    
    new_homographies = []
    all_corners = []
    
    # Recalculate all homographies relative to the CENTER image
    for i, H in enumerate(homographies):
        H_new = H_0_to_mid.dot(H) 
        new_homographies.append(H_new)
        
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_new))

    # Determine canvas size
    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    out_w, out_h = x_max - x_min, y_max - y_min
    
    print(f"   Canvas Size: {out_w} x {out_h}")
    
    # 3. Stitch and Blend
    # Translation matrix to shift negative coordinates to positive
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Draw order: From extremities inwards to the center
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - mid_idx), reverse=True)
    
    print("[Scratch] Blending images...")
    for i in draw_order:
        H_final = H_translation.dot(new_homographies[i])
        warped = cv2.warpPerspective(images[i], H_final, (out_w, out_h), flags=cv2.INTER_LINEAR)
        
        # Create mask and erode edges to hide seams
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1) 
        
        # Overlay image
        panorama[mask > 0] = warped[mask > 0]
        
    return crop_black_borders(panorama)

def run_stitching_opencv(images):
    """
    OpenCV logic UPDATED to use Affine transformation (Partial 2D)
    instead of Homography to avoid bowtie distortion.
    """
    print(f"\n--- Starting OpenCV Stitching on {len(images)} Images ---")
    homographies = [np.identity(3)]
    
    for i in range(len(images) - 1):
        sift = cv2.SIFT_create()
        kp_p, desc_p = sift.detectAndCompute(images[i], None)
        kp_c, desc_c = sift.detectAndCompute(images[i+1], None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_c, desc_p, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: good.append(m)
        
        if len(good) < 4: 
            print(f"[OpenCV] Not enough matches between Image {i+1} and {i+2}")
            return None

        src_pts = np.float32([kp_c[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_p[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # --- FIX: Use Affine Partial (Rotation + Scale + Trans) ---
        # This prevents the extreme stretching found in Homography chains
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is None:
            print(f"[OpenCV] Could not calculate Affine transform between Image {i+1} and {i+2}")
            return None
        
        # Convert 2x3 Affine to 3x3 Homography matrix
        H = np.vstack([M, [0, 0, 1]])
            
        homographies.append(homographies[i].dot(H))
        
    mid_idx = len(images) // 2
    H_0_to_mid = np.linalg.inv(homographies[mid_idx])
    
    new_homographies = [H_0_to_mid.dot(H) for H in homographies]
    
    all_corners = []
    for i, H in enumerate(new_homographies):
        h, w = images[i].shape[:2]
        c = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        all_corners.append(cv2.perspectiveTransform(c, H))
        
    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    out_w, out_h = x_max - x_min, y_max - y_min
    
    panorama = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    H_trans = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - mid_idx), reverse=True)
    for i in draw_order:
        H_final = H_trans.dot(new_homographies[i])
        warped = cv2.warpPerspective(images[i], H_final, (out_w, out_h))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
        panorama[mask > 0] = warped[mask > 0]
        
    return crop_black_borders(panorama)

def main():
    relative_img_dir = input("Enter path to image folder: ").strip()
    image_dir = os.path.abspath(relative_img_dir)
    if not os.path.exists(image_dir):
        print("Folder not found.")
        return

    # Load and Resize
    # Added print to verify order
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")), key=lambda s: natural_sort_key(s))
    print(f"Loading {len(image_paths)} images in order:")
    for p in image_paths:
        print(f" - {os.path.basename(p)}")

    original_images = [cv2.imread(p) for p in image_paths if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
    original_images = [img for img in original_images if img is not None]
    
    if len(original_images) < 2:
        print("Need at least 2 images.")
        return

    # Resize for Custom SIFT speed
    images = [resize_image(img, width=WORK_WIDTH) for img in original_images]

    # Run Custom Implementation
    res_custom = run_stitching_custom(images)
    
    # Run OpenCV Implementation
    res_opencv = run_stitching_opencv(images)

    # Save results without opening any GUI windows
    custom_output = os.path.join(image_dir, "stitched_custom_affine.jpg")
    opencv_output = os.path.join(image_dir, "stitched_opencv_affine.jpg")

    if res_custom is not None:
        cv2.imwrite(custom_output, res_custom)
        print(f"Custom stitch saved to {custom_output}")
    else:
        print("Custom stitching failed; no image saved.")

    if res_opencv is not None:
        cv2.imwrite(opencv_output, res_opencv)
        print(f"OpenCV stitch saved to {opencv_output}")
    else:
        print("OpenCV stitching failed; no image saved.")

if __name__ == "__main__":
    main()