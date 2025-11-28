"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 3 Assignment Part 1 - Tasks 1-3 (Gradients, Edge/Corner Detection, Boundary Detection

The purpose of this script is to create gradient of the images 
(Laplacian of Gaussian), detect edges and corners in the images, 
and determine the exact boundaries of objects

Usage:
    1. Run the script - python module3_part1.py
    2. Select the task number once prompted
    3. Provide the folder path with the 10 images
    4. Once the images are rendered in a separate pop up window, press :-
        a. Spacebar to move to the next image
        b. s to Save
        c. q to Quit

Main dependency is numpy
"""

import cv2
import numpy as np
import glob
import os

# Task 1 functionality
def process_gradients_log(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    # Using Sobel (kernel size 3) from OpenCV
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3) # Gradient calculation (x direction)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3) # Gradient calculation (y direction)
    magnitude = cv2.magnitude(gradient_x, gradient_y) # Edge strength
    angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True) # Edge direction

    # Normalization for visualization
    magnitude_visualization = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    angle_visualization = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Laplacian of Gaussian filtered versions
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    laplacian_of_gaussian = cv2.Laplacian(blurred, cv2.CV_64F)
    # Normalization of Laplacian of Gaussian
    laplacian_of_gaussian_visualization = cv2.normalize(np.abs(laplacian_of_gaussian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Creating canvas to display all the results together
    height, width = grayscale.shape
    canvas = np.zeros((height*2, width*2, 3), dtype=np.uint8)
    
    # BGR conversion from grayscale
    magnitude_bgr = cv2.cvtColor(magnitude_visualization, cv2.COLOR_GRAY2BGR)
    angle_bgr = cv2.cvtColor(angle_visualization, cv2.COLOR_GRAY2BGR)
    laplacian_of_gaussian_bgr = cv2.cvtColor(laplacian_of_gaussian_visualization, cv2.COLOR_GRAY2BGR)

    # Arranging images into a 2 x 2 grid layout
    canvas[0:height, 0:width] = image
    canvas[0:height, width:2*width] = magnitude_bgr
    canvas[height:2*height, 0:width] = angle_bgr
    canvas[height:2*height, width:2*width] = laplacian_of_gaussian_bgr
    # Text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original Image", (10, 30), font, 0.7, (0,0,255), 2)
    cv2.putText(canvas, "Gradient Magnitude", (width+10, 30), font, 0.7, (0,0,255), 2)
    cv2.putText(canvas, "Gradient Angle", (10, height+30), font, 0.7, (0,0,255), 2)
    cv2.putText(canvas, "Laplacian of Gaussian", (width+10, height+30), font, 0.7, (0,0,255), 2)
    return canvas

# Task 2 functionality
def detect_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    output = image.copy()
    # Using Sobel (kernel size 3) from OpenCV
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3) # Gradient calculation (x direction)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3) # Gradient calculation (y direction)
    magnitude = cv2.magnitude(gradient_x, gradient_y) # Edge strength
    edge_threshold = np.percentile(magnitude, 85)
    edges = magnitude > edge_threshold
    output[edges] = [0, 255, 0]
    # Using absolute values for corner detection
    gradient_x = np.abs(gradient_x) 
    gradient_y = np.abs(gradient_y)
    
    # Normalization process
    gradient_x_normalized = cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gradient_y_normalized = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # We are using top 20% instead of top 15% so more corners are detected
    gradient_x_threshold = np.percentile(gradient_x_normalized, 80)
    gradient_y_threshold = np.percentile(gradient_y_normalized, 80)
    # Gradients x and gradients y have to be above their respective thresholds which will distinguish corners from edges
    corners = (gradient_x_normalized > gradient_x_threshold) & (gradient_y_normalized > gradient_y_threshold)
    corners = corners & ~edges # Making sure edges aren't marked as corners
    output[corners] = [0, 0, 255]
    cv2.putText(output, "Edges (green) :- higher gradient magnitude", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, "Corners (red) :- higher gradient x and gradient y", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output

# Task 3 functionality
def find_exact_boundary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Gaussian blur application
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold step
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Getting contours from binary image
    output = image.copy()

    if contours:
        c = max(contours, key=cv2.contourArea) # Assuming largest contour is object of interest
        cv2.drawContours(output, [c], -1, (0, 255, 0), 3)
    return output

# Main function
def main():
    images = []    
    print("Module 3 (tasks 1- 3)")
    print("Please select one of the following options")
    print("1. Gradients & Laplacian of Gaussian")
    print("2. Keypoints detection (edges & corners)")
    print("3. Exact boundary detection")
    
    # Make sure the user selection is valid
    while True:
        try:
            task_input = input("Enter Task (numbers 1, 2, or 3 are valid selections) :- ").strip()
            if not task_input: continue # Empty input case
            task_number = int(task_input)
            if 1 <= task_number <= 3: break# Valid task number was entered by user
            else: print("Please enter numbers 1, 2, or 3")
        except ValueError: print("Please enter numbers 1, 2, or 3") 
    
    while True:
        src = input("Please enter the image / folder path :- ").strip()
        if os.path.exists(src):
            if os.path.isfile(src): # Single image case
                images = [src]
            else: # Folder / multiple images case (main one)
                files = sorted(glob.glob(os.path.join(src, "*.*")))
                images = [x for x in files if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
                if not images: # If folder did not contain any valid image files
                    print("Folder does not contain any valid image files")
                    continue
            break
        else: 
            print("Path to image / folder was invalid. Please make sure to enter a correct path")

    print(f"\nRunning the task number {task_number}")
    print("Press spacebar to move to next picture, s to save, and q to quit program")

    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is None: continue # Skip any corrupt frames
        # Routing the image to the proper function based on the user input
        if task_number == 1: result = process_gradients_log(frame)
        elif task_number == 2: result = detect_features(frame)
        elif task_number == 3: result = find_exact_boundary(frame)
        cv2.imshow(f"Task number {task_number}", result) # Display window
        
        stop = False
        while True:
            # Wait for the key selection
            k = cv2.waitKey(0) & 0xFF            
            if k == 32: break  # If spacebar, move to next image
            if k == ord('q'): stop = True; break  # q means quit the program
            if k == ord('s'): # s means save the result
                out_name = f"out_task{task_number}_{os.path.basename(image_path)}"
                cv2.imwrite(out_name, result)
                print(f"File has been saved to {out_name}")
        
        if stop: break

    # Clean up all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()