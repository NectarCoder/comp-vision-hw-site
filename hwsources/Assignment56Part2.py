"""
CSC 8830 Helper Script
Description: 
    Generates a dummy .npz file containing binary masks.
    This allows you to test the "SAM2" mode of the main assignment script
    without needing to run the actual SAM2 model (which is heavy).
    
    It creates a moving circle mask simulating an object moving across 100 frames.
"""

import numpy as np
import cv2

def generate_data():
    masks = {}
    num_frames = 300
    width, height = 640, 480
    
    print(f"Generating masks for {num_frames} frames...")
    
    for i in range(num_frames):
        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Simulate a moving object (Circle bouncing)
        x = int(50 + (i * 5) % (width - 100))
        y = int(240 + 100 * np.sin(i * 0.1))
        radius = 50
        
        cv2.circle(mask, (x, y), radius, 1, -1) # 1 for binary mask
        
        key = f"frame_{i}"
        masks[key] = mask

    # Save to NPZ
    filename = "sam2_masks.npz"
    np.savez_compressed(filename, **masks)
    print(f"Successfully saved {filename}.")
    print("Place this file in the same directory as 'assignment5_tracking_suite.py'.")

if __name__ == "__main__":
    generate_data()