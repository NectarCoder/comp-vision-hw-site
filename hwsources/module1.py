"""
CSC 8830 Assignment 1 - Dimension Calculator Script
Author: [Your Name]

Description:
    This script calculates the real-world dimensions of an object using 
    perspective projection equations. It requires a calibration step to 
    determine the camera's focal length (in pixels) or a manual entry 
    of a known focal length.

Dependencies:
    opencv-python (pip install opencv-python)
    numpy (pip install numpy)

Usage:
    1. Run the script: python dimension_calculator.py
    2. Enter the image filename when prompted.
    3. Click two points on the image to define the dimension to measure.
    4. Follow the console prompts to Calibrate or Measure.
    5. Press 'r' to reset points, 'q' to quit.

Theory:
    Based on the Pinhole Camera Model:
    w / f = W / D
    
    Where:
    w = Object width in pixels (measured from image)
    f = Focal length in pixels (internal camera parameter)
    W = Real-world object width
    D = Distance from camera to object

    Therefore:
    W = (w * D) / f
"""

import cv2
import math
import numpy as np
import os

# --- Global Variables ---
points = []
image = None
clone = None
window_name = "Dimension Calculator"

def click_event(event, x, y, flags, param):
    """
    Mouse callback function to capture point clicks.
    """
    global points, image, clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            # Reset if already have 2 points
            points = []
            image = clone.copy()
            cv2.imshow(window_name, image)

        points.append((x, y))
        
        # Visual feedback: Draw a circle where clicked
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        if len(points) == 2:
            # Draw a line between the two points
            cv2.line(image, points[0], points[1], (0, 255, 0), 2)
            dist_px = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
            print(f"\n[INFO] Points selected: {points}")
            print(f"[INFO] Pixel Distance (w): {dist_px:.2f} pixels")
            process_measurement(dist_px)
            
        cv2.imshow(window_name, image)

def process_measurement(pixel_width):
    """
    Handles the logic for Calibration vs Measurement based on user input.
    """
    print("\n--- ACTION MENU ---")
    print("1. Calibrate (Calculate Focal Length)")
    print("2. Measure (Calculate Real Dimension)")
    print("Press any other key to cancel/re-select.")
    
    # Note: In a pure GUI app, these would be buttons. 
    # For a CV script, console input is standard for parameters.
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        try:
            real_width = float(input("Enter REAL width of the object (e.g., cm, in): "))
            distance = float(input("Enter DISTANCE from camera to object (same units as width): "))
            
            # Formula: f = (w * D) / W
            focal_length = (pixel_width * distance) / real_width
            print(f"\n[RESULT] Calculated Focal Length (f): {focal_length:.2f} pixels")
            print("KEEP THIS VALUE SAFE for measuring other objects at known distances!")
        except ValueError:
            print("[ERROR] Invalid numeric input.")
            
    elif choice == '2':
        try:
            focal_length = float(input("Enter known Focal Length (pixels): "))
            distance = float(input("Enter DISTANCE from camera to object: "))
            
            # Formula: W = (w * D) / f
            real_dim = (pixel_width * distance) / focal_length
            
            print(f"\n[RESULT] REAL WORLD DIMENSION: {real_dim:.4f} units")
            
            # Draw result on image
            cv2.putText(image, f"{real_dim:.2f} units", 
                       (points[0][0], points[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, image)
            
        except ValueError:
            print("[ERROR] Invalid numeric input.")
    else:
        print("[INFO] Action cancelled. Click new points to retry.")

def main():
    global image, clone
    
    print("=== Dimension Calculator Started ===")
    
    # 1. Load Image
    user_input = input("Enter image filename/path (e.g., test.jpg): ").strip()

    # Expand user and convert to an absolute or relative path.
    img_path = os.path.expanduser(user_input)

    # If the path doesn't exist in the current working directory, try
    # resolving it relative to this script's directory (useful when the
    # image sits next to the script but the cwd is the project root).
    if not os.path.exists(img_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, img_path)
        if os.path.exists(alt_path):
            img_path = alt_path
        else:
            print(f"[ERROR] File '{user_input}' not found.")
            return

    original_image = cv2.imread(img_path)
    if original_image is None:
        print("[ERROR] Could not decode image.")
        return

    # Resize for display if image is too massive
    height, width = original_image.shape[:2]
    max_height = 800
    if height > max_height:
        scale = max_height / height
        image = cv2.resize(original_image, (int(width * scale), int(height * scale)))
    else:
        image = original_image.copy()

    clone = image.copy()

    # 2. Setup Window and Mouse Callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    print("\n[INSTRUCTIONS]")
    print("- Click TWO points on the image to define the object width.")
    print("- Check the console for inputs after clicking.")
    print("- Press 'r' to clear lines.")
    print("- Press 'q' to exit.")

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            global points
            points = []
            print("[INFO] Reset.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()