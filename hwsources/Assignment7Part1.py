"""
CSC 8830 Assignment 7 - Problem 1: Calibrated Stereo Size Estimation
Author: [Your Name]

Description:
    Calculates the real-world size of objects using Calibrated Stereo Vision.
    1. Loads Left and Right stereo images.
    2. Computes Disparity Map using StereoSGBM.
    3. Calculates Depth (Z) using the formula: Z = (f * B) / disparity
    4. Calculates real-world 3D coordinates (X, Y, Z) for clicked points.
    5. Computes Euclidean distance between two 3D points.

Usage:
    python stereo_size_estimator.py
    (Follow prompts to enter image paths and calibration parameters)
"""

import cv2
import numpy as np

# --- Global Variables ---
window_name = "Stereo Depth & Measurement"
points = []
current_disparity_map = None
Q_matrix = None # Reprojection matrix
focal_length = 0
baseline = 0
imgL_display = None

def select_points(event, x, y, flags, param):
    global points, imgL_display, current_disparity_map, focal_length, baseline

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get disparity value at clicked point
        # Note: Disparity map usually computed at smaller scale or different type
        if current_disparity_map is None: return

        disp_val = current_disparity_map[y, x]
        
        if disp_val <= 0:
            print(f"[WARNING] Invalid disparity at ({x},{y}). Try clicking a textured area.")
            return

        # 1. Calculate Depth Z
        # Z = (f * B) / disparity
        # Note: StereoSGBM output is usually scaled by 16
        true_disp = disp_val / 16.0
        if true_disp == 0: return 
        
        Z = (focal_length * baseline) / true_disp

        # 2. Calculate X, Y using Pinhole model
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # For simplicity, assuming center of image is principal point (cx, cy)
        h, w = imgL_display.shape[:2]
        cx, cy = w / 2, h / 2
        
        X = (x - cx) * Z / focal_length
        Y = (y - cy) * Z / focal_length

        coord_3d = (X, Y, Z)
        points.append(coord_3d)

        # Visual Feedback
        cv2.circle(imgL_display, (x, y), 5, (0, 255, 0), -1)
        print(f"[POINT] Pixel: ({x}, {y}) | Disparity: {true_disp:.2f} | 3D Coord: ({X:.2f}, {Y:.2f}, {Z:.2f})")

        if len(points) == 2:
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            
            # Calculate Euclidean Distance in 3D space
            dist = np.linalg.norm(p1 - p2)
            
            print(f"--> [RESULT] Real World Size/Distance: {dist:.4f} units")
            
            # Draw line and text
            pt1_2d = (int(p1[0] * focal_length / p1[2] + cx), int(p1[1] * focal_length / p1[2] + cy))
            pt2_2d = (int(p2[0] * focal_length / p2[2] + cx), int(p2[1] * focal_length / p2[2] + cy))
            
            cv2.line(imgL_display, pt1_2d, pt2_2d, (0, 0, 255), 2)
            mid_x = int((pt1_2d[0] + pt2_2d[0]) / 2)
            mid_y = int((pt1_2d[1] + pt2_2d[1]) / 2)
            cv2.putText(imgL_display, f"{dist:.2f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            points = [] # Reset for next measurement

        cv2.imshow(window_name, imgL_display)

def main():
    global imgL_display, current_disparity_map, focal_length, baseline
    print("=== Calibrated Stereo Measurement ===")
    
    # 1. Configuration (You should calibrate your camera to get these real values)
    # Defaults provided for testing
    f_input = input("Enter Focal Length in pixels (default 700): ").strip()
    focal_length = float(f_input) if f_input else 700.0
    
    b_input = input("Enter Baseline distance in units (e.g., cm) (default 5.0): ").strip()
    baseline = float(b_input) if b_input else 5.0

    pathL = input("Path to LEFT image: ").strip()
    pathR = input("Path to RIGHT image: ").strip()
    
    # Load images
    imgL = cv2.imread(pathL)
    imgR = cv2.imread(pathR)

    if imgL is None or imgR is None:
        print("[ERROR] Could not load images.")
        return

    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 2. Compute Disparity (StereoSGBM is better than StereoBM)
    print("[INFO] Computing Disparity Map...")
    window_size = 3
    min_disp = 0
    num_disp = 16 * 5 # Must be divisible by 16
    
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        P1 = 8 * 3 * window_size**2,
        P2 = 32 * 3 * window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    disp = stereo.compute(grayL, grayR).astype(np.float32)
    current_disparity_map = disp
    
    # Normalize disparity for visualization
    disp_vis = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    imgL_display = imgL.copy()
    
    # 3. Interactive Measurement
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_points)
    
    print("\n[INSTRUCTIONS]")
    print("- Left Image is shown.")
    print("- Click on TWO points (edges of object).")
    print("- Script will calculate depth Z and then physical size.")
    print("- 'd' to toggle Disparity View.")
    print("- 'q' to quit.")

    show_disp = False

    while True:
        if show_disp:
            cv2.imshow(window_name, disp_vis)
        else:
            cv2.imshow(window_name, imgL_display)
            
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('d'):
            show_disp = not show_disp

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()