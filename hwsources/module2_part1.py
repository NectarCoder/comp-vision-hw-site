"""
CSC 8830 Assignment 2 - Problem 1: Template Matching
Author: [Your Name]

Description:
    This script demonstrates object detection using Template Matching via 
    Normalized Cross-Correlation (cv2.TM_CCOEFF_NORMED).
    It is designed to evaluate multiple templates against a target scene.

Usage:
    1. Organize your templates in a folder (e.g., ./templates/).
    2. Have a target scene image ready.
    3. Run: python template_matching_demo.py
"""

import cv2
import numpy as np
import os
import glob

def main():
    print("=== Template Matching Object Detector ===")
    
    # 1. Get Input Paths
    scene_path = input("Enter path to the Scene Image (e.g., scene.jpg): ").strip()
    template_dir = input("Enter directory containing Template images (e.g., ./templates): ").strip()
    
    if not os.path.exists(scene_path):
        print(f"[ERROR] Scene image '{scene_path}' not found.")
        return
    
    if not os.path.isdir(template_dir):
        print(f"[ERROR] Template directory '{template_dir}' not found.")
        return

    # Load Scene
    img_rgb = cv2.imread(scene_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Get list of template files
    template_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        template_files.extend(glob.glob(os.path.join(template_dir, ext)))
    
    print(f"[INFO] Found {len(template_files)} templates.")
    
    # Threshold for detection (Correlation Coefficient)
    # 0.8 is a strong match, adjust lower if lighting varies significantly
    threshold = 0.8 
    
    detections = 0
    
    # Copy of image to draw rectangles on
    output_image = img_rgb.copy()

    for t_file in template_files:
        template = cv2.imread(t_file, 0) # Load as grayscale
        if template is None:
            continue
            
        w, h = template.shape[::-1]
        
        # Apply Template Matching
        # TM_CCOEFF_NORMED is robust to lighting changes
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where correlation is above threshold
        loc = np.where(res >= threshold)
        
        # If matches found
        if len(loc[0]) > 0:
            print(f"[MATCH] Found match for {os.path.basename(t_file)}")
            
            # Draw rectangles
            # We zip(*loc[::-1]) to iterate over (x, y) points
            for pt in zip(*loc[::-1]):
                cv2.rectangle(output_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                cv2.putText(output_image, os.path.basename(t_file), 
                           (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            detections += 1
        else:
            print(f"[INFO] No match for {os.path.basename(t_file)}")

    print(f"\n[INFO] Detection complete. Objects detected from {detections} templates.")
    
    # Show result
    cv2.imshow("Detections", output_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()