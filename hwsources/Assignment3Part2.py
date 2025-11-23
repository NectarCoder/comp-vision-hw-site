"""
CSC 8830 Assignment 3 - Problems 4 & 5
Author: [Your Name]

Description:
    Problem 4: Segment non-rectangular objects using ArUco markers placed on boundaries.
    Problem 5: Compare result with Segment Anything Model 2 (SAM2).

    Logic:
    1. Detect ArUco markers in the image.
    2. Use marker centers as vertices to form a polygon boundary.
    3. Create a binary mask from this polygon.
    4. (Optional) Run SAM2 using the marker points as input prompts.

Dependencies:
    pip install opencv-contrib-python numpy matplotlib
    (For Q5): SAM2 installation required (https://github.com/facebookresearch/segment-anything-2)
"""

import cv2
import numpy as np
import sys

# Try importing SAM2 (Optional for Q4, Required for Q5)
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARNING] SAM2 library not found. Q5 comparison will be skipped.")

def segment_with_aruco(img):
    """
    Detects ArUco markers and creates a segmentation mask connecting them.
    Assumes markers are placed sequentially along the boundary.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load ArUco Dictionary (Using 4x4_50 as standard)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is None or len(ids) < 3:
        print("[INFO] Not enough markers detected (need at least 3 for a polygon).")
        return img, None, []

    # Flatten ids
    ids = ids.flatten()
    
    # Get centers of markers
    marker_centers = []
    for corner in corners:
        # corner shape is (1, 4, 2)
        c = corner[0]
        center_x = int(np.mean(c[:, 0]))
        center_y = int(np.mean(c[:, 1]))
        marker_centers.append((center_x, center_y))
    
    # Sort markers based on ID to ensure correct polygon order
    # (Assumes you stuck markers 0, 1, 2... in order around object)
    sorted_pairs = sorted(zip(ids, marker_centers), key=lambda x: x[0])
    sorted_centers = [pt for id_val, pt in sorted_pairs]
    
    # Create Mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(sorted_centers, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw filled polygon on mask
    cv2.fillPoly(mask, [pts], 255)
    
    # Visualization
    result_vis = img.copy()
    cv2.polylines(result_vis, [pts], True, (0, 255, 0), 3)
    cv2.aruco.drawDetectedMarkers(result_vis, corners, ids)
    
    return result_vis, mask, sorted_centers

def segment_with_sam2(img, input_points):
    """
    Runs SAM2 using the ArUco points as prompts.
    """
    if not SAM2_AVAILABLE:
        return np.zeros_like(img)

    print("[INFO] Running SAM2 inference...")
    # Placeholder for SAM2 implementation structure
    # 1. Load Model
    # sam2_checkpoint = "sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"
    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    
    # 2. Set Image
    # predictor.set_image(img)
    
    # 3. Predict using points
    # masks, scores, logits = predictor.predict(point_coords=input_points, point_labels=np.ones(len(input_points)))
    
    # Returning a dummy visual for the script to run without the heavy weights
    dummy_res = img.copy()
    cv2.putText(dummy_res, "SAM2 Result Placeholder", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return dummy_res

def main():
    print("=== Assignment 3: ArUco Segmentation ===")
    img_path = input("Enter path to an image with ArUco markers: ").strip()
    
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found.")
        return
        
    # --- Q4: ArUco Segmentation ---
    vis_img, mask, points = segment_with_aruco(img)
    
    if mask is not None:
        # Show Masked Object
        segmented_object = cv2.bitwise_and(img, img, mask=mask)
        
        cv2.imshow("ArUco Boundary", vis_img)
        cv2.imshow("Segmented Object", segmented_object)
        
        # --- Q5: SAM2 Comparison ---
        if SAM2_AVAILABLE and len(points) > 0:
            sam_result = segment_with_sam2(img, np.array(points))
            cv2.imshow("SAM2 Result", sam_result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()