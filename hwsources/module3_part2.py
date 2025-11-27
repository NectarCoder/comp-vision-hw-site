"""
CSc 8830: Computer Vision - Assignment 3 (Part 2)
File: module3_part2.py

Description:
    Implements Problems 4 & 5:
    1. Task 4: Segment non-rectangular objects using ArUco markers (Ground Truth).
       - Logic: Detects markers, sorts them by ID (0->1->2...), and fills the polygon.
    2. Task 5: Deep Learning Comparison (SAM2).
       - Logic: Uses the ArUco points as a prompt for SAM2 and compares IoU.

Usage:
    python module3_part2.py
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# --- Check for SAM2 Availability ---
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARNING] SAM2 library not found. Comparison (Task 5) will be skipped.")
    print("To install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

# ==========================================
#  CORE FUNCTIONS
# ==========================================

def segment_with_aruco(img):
    """
    Task 4: Detects ArUco markers and creates a segmentation mask.
    Crucial: Sorts points by Marker ID to ensure the polygon connects correctly.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load ArUco Dictionary (Standard 4x4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    sorted_centers = []

    if ids is not None and len(ids) >= 3:
        # Flatten IDs for zipping
        ids_flat = ids.flatten()
        
        # Calculate center point for each marker
        marker_centers = []
        for corner in corners:
            c = corner[0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            marker_centers.append((center_x, center_y))
        
        # --- ID-Based Sorting ---
        # We sort the points based on the Marker ID (0, 1, 2, 3...)
        # This allows you to define complex/concave shapes by placing markers in order.
        sorted_pairs = sorted(zip(ids_flat, marker_centers), key=lambda x: x[0])
        sorted_centers = [pt for _, pt in sorted_pairs]
        
        # Create Polygon Mask (White on Black)
        pts = np.array(sorted_centers, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
        # Visualization Image (Green Outline)
        vis_img = img.copy()
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)
        cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
        
        return vis_img, mask, np.array(sorted_centers)
    
    else:
        print(f"[INFO] Found {0 if ids is None else len(ids)} markers. Need at least 3.")
        return img, mask, []

def segment_with_sam2(img, input_points):
    """
    Task 5: Runs SAM2 inference using ArUco points as the input prompt.
    """
    if not SAM2_AVAILABLE:
        # Return empty mask if SAM2 isn't installed
        return np.zeros(img.shape[:2], dtype=np.uint8)

    # --- SAM2 Configuration ---
    # NOTE: You must download these files from the SAM2 repository
    checkpoint = "checkpoints/sam2_hiera_large.pt" 
    model_cfg = "sam2_hiera_l.yaml"
    
    # Check if model exists
    if not os.path.exists(checkpoint):
        # Fallback to looking in current directory
        if os.path.exists("sam2_hiera_large.pt"):
            checkpoint = "sam2_hiera_large.pt"
        else:
            print(f"[ERROR] SAM2 Checkpoint not found at '{checkpoint}'.")
            print("Please download 'sam2_hiera_large.pt' and place it in the folder.")
            return np.zeros(img.shape[:2], dtype=np.uint8)

    print("[INFO] Loading SAM2 Model (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set Image (Convert BGR to RGB for SAM2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # --- Prompting SAM2 ---
    # We use the Bounding Box of the ArUco markers as the prompt
    x_min, y_min = np.min(input_points, axis=0)
    x_max, y_max = np.max(input_points, axis=0)
    box_prompt = np.array([x_min, y_min, x_max, y_max])

    print(f"[INFO] Predicting with Box Prompt: {box_prompt}")
    masks, scores, _ = predictor.predict(
        box=box_prompt,
        multimask_output=False
    )
    
    # Return binary mask (0 or 255)
    sam_mask = (masks[0] * 255).astype(np.uint8)
    return sam_mask

def calculate_iou(mask1, mask2):
    """ Calculate Intersection over Union Score """
    # Convert to boolean (True/False)
    m1 = mask1 > 0
    m2 = mask2 > 0
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0: return 0.0
    return intersection / union

# ==========================================
#  MAIN EXECUTION
# ==========================================
def main():
    print("\n=== CSc 8830: Assignment 3 - Part 2 (Tasks 4 & 5) ===")
    
    # 1. Get Image Path
    img_path = input("Enter path to image (containing ArUco markers): ").strip()
    if not os.path.exists(img_path):
        print("File not found.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image.")
        return
    
    # 2. Run ArUco Segmentation (Ground Truth)
    print("\n--- Running Task 4: ArUco Segmentation ---")
    vis_aruco, mask_aruco, points = segment_with_aruco(img)
    
    if len(points) == 0:
        print("Comparison aborted: No markers found.")
        return

    # 3. Run SAM2 Segmentation (AI Prediction)
    print("\n--- Running Task 5: SAM2 Comparison ---")
    mask_sam = segment_with_sam2(img, points)

    # 4. Compare (IoU)
    if np.max(mask_sam) > 0:
        iou = calculate_iou(mask_aruco, mask_sam)
        print(f"\n[RESULT] Intersection over Union (IoU) Score: {iou:.4f}")
    else:
        iou = 0.0
        print("\n[RESULT] SAM2 did not generate a mask (or is not installed). IoU: 0.0")

    # 5. Visualization using Matplotlib
    plt.figure(figsize=(15, 6))
    
    # Plot 1: ArUco (Ground Truth)
    plt.subplot(1, 3, 1)
    plt.title("Task 4: ArUco Mask (Ground Truth)")
    plt.imshow(mask_aruco, cmap='gray')
    plt.axis('off')

    # Plot 2: SAM2 (Prediction)
    plt.subplot(1, 3, 2)
    plt.title("Task 5: SAM2 Mask (AI Prediction)")
    plt.imshow(mask_sam, cmap='gray')
    plt.axis('off')

    # Plot 3: Overlay / Comparison
    plt.subplot(1, 3, 3)
    plt.title(f"Overlay (IoU: {iou:.2f})")
    
    # Create colored overlay
    # Ground Truth (ArUco) = Green
    # Prediction (SAM2)    = Red
    # Overlap              = Yellow
    overlay = np.zeros_like(img)
    overlay[mask_aruco > 0] = [0, 255, 0] 
    # Add SAM2 in Red (using logical OR to keep colors distinct if needed, or simply overwrite)
    # To see overlap clearly, we add pixel values
    
    # Reset for cleaner overlay logic using float
    vis_overlay = img.copy().astype(float)
    
    # Green channel boost for ArUco
    vis_overlay[:, :, 1] = np.where(mask_aruco > 0, vis_overlay[:, :, 1] + 100, vis_overlay[:, :, 1])
    # Red channel boost for SAM2
    vis_overlay[:, :, 2] = np.where(mask_sam > 0, vis_overlay[:, :, 2] + 100, vis_overlay[:, :, 2])
    
    vis_overlay = np.clip(vis_overlay, 0, 255).astype(np.uint8)

    plt.imshow(cv2.cvtColor(vis_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Optional: Save Result
    save_q = input("\nSave result image? (y/n): ").strip().lower()
    if save_q == 'y':
        out_name = "comparison_result.png"
        plt.savefig(out_name)
        print(f"Saved to {out_name}")

if __name__ == "__main__":
    main()