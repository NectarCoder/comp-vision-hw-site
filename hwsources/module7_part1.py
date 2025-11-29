#TODO Add description
import cv2
import math
import os
import numpy as np


# --- SHARED COMPUTATION HELPERS ---

def compute_focal_length_from_reference(pixel_width: float, real_width_cm: float, distance_cm: float) -> float:
    if real_width_cm == 0:
        raise ValueError("Real-world width must be non-zero")
    return (pixel_width * distance_cm) / real_width_cm


def compute_depth_from_stereo(focal_length: float, baseline_cm: float, disparity_px: float) -> float:
    if disparity_px == 0:
        raise ValueError("Disparity must be non-zero")
    return (focal_length * baseline_cm) / disparity_px


def compute_real_size_from_depth(pixel_distance: float, focal_length: float, depth_cm: float) -> float:
    if focal_length == 0:
        raise ValueError("Focal length must be non-zero")
    return (pixel_distance * depth_cm) / focal_length

# --- UTILITY FUNCTIONS ---

def prompt_float(prompt_text):
    """Helper to safely get float input from user."""
    while True:
        try:
            val = input(prompt_text).strip()
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_image(prompt_text, max_height=800):
    """Prompts user for path, loads image, and resizes for screen fitting."""
    while True:
        path = input(prompt_text).strip()
        # Handle home directory expansion and absolute paths
        path = os.path.abspath(os.path.expanduser(path))
        
        if not os.path.exists(path):
            print(f"Error: File not found at {path}. Please try again.")
            continue
            
        img = cv2.imread(path)
        if img is None:
            print("Error: Not a valid image file. Please try again.")
            continue
            
        # Resize if too big for screen
        h, w = img.shape[:2]
        if h > max_height:
            scale = max_height / h
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        print(f"Successfully loaded: {os.path.basename(path)}")
        return img

def draw_text(img, text, pos, color=(0, 255, 0)):
    """Draws text with a black outline for visibility."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Outline
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)     # Text

# --- CORE LOGIC BLOCKS ---

def get_focal_length():
    """Phase 1: Calculates Focal Length from Reference Image."""
    print("\n" + "="*50)
    print("PHASE 1: CAMERA CALIBRATION")
    print("WARNING: Reference image must be taken with the SAME CAMERA as test images.")
    print("="*50)

    ref_img = load_image("\nEnter path to REFERENCE Image: ")
    real_width = prompt_float("Enter Real Width of the reference object (cm): ")
    real_dist = prompt_float("Enter Real Distance from camera (cm): ")

    print("\nINSTRUCTIONS:")
    print("1. A window will open.")
    print("2. Click the LEFT edge of the object.")
    print("3. Click the RIGHT edge of the object.")
    print("   (Window will close automatically 1 second after selection)")
    print("   (Press 'r' to reset points if you make a mistake)")
    
    # Selection Loop
    points = []
    clone = ref_img.copy()
    window_name = "Calibration: Select Width"
    
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                # Draw line immediately when 2nd point is clicked
                if len(points) == 2:
                    cv2.line(clone, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)
    cv2.imshow(window_name, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 2:
            # Short pause so user can see the line before window closes
            cv2.waitKey(1000)
            break
        if key == ord('r'):
            points = []
            clone = ref_img.copy()
            cv2.imshow(window_name, clone)
            
    cv2.destroyAllWindows()

    pixel_width = math.dist(points[0], points[1])
    focal_length = compute_focal_length_from_reference(pixel_width, real_width, real_dist)
    
    print(f"\n[CALIBRATION RESULT]")
    print(f"Pixel Width: {pixel_width:.2f} px")
    print(f"Calculated Focal Length: {focal_length:.2f}")
    return focal_length

def get_stereo_depth(focal_length):
    """Phase 2: Calculates Z (Depth) using Stereo Pair."""
    print("\n" + "="*50)
    print("PHASE 2: STEREO DEPTH ESTIMATION")
    print("WARNING: Images must be Left/Right pair from SAME CAMERA.")
    print("="*50)

    imgL = load_image("Enter path to LEFT Image: ")
    imgR = load_image("Enter path to RIGHT Image: ")
    baseline = prompt_float("Enter Baseline (camera movement distance in cm): ")

    # Combine images for easier matching
    h, w = imgL.shape[:2]
    # Ensure heights match for stacking
    if imgR.shape[0] != h:
        imgR = cv2.resize(imgR, (int(imgR.shape[1] * (h / imgR.shape[0])), h))
    
    vis = np.hstack((imgL, imgR))
    
    print("\nINSTRUCTIONS:")
    print("1. Click a distinct point in the LEFT image (Left side of window).")
    print("2. Click the SAME distinct point in the RIGHT image (Right side of window).")
    print("   (Window will close automatically 1 second after selection)")
    
    points = []
    window_name = "Stereo: Match Points (Left -> Right)"
    
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
                cv2.imshow(window_name, vis)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)
    cv2.imshow(window_name, vis)
    
    # Wait for 2 points
    while len(points) < 2:
        if cv2.waitKey(1) & 0xFF == ord('q'): # Emergency exit
            break
    
    # Short pause to see the points before closing
    if len(points) == 2:
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
    
    if len(points) < 2:
        return None, None

    pL, pR = points[0], points[1]
    
    # Logic to handle user clicking Right image first by accident, or normal Left-first
    # The split line is at x = w
    x1, x2 = pL[0], pR[0]
    
    # Identify which click belongs to which image side
    if x1 < w and x2 >= w:
        # Correct order: Click 1 is Left, Click 2 is Right
        disp_x_left = x1
        disp_x_right = x2 - w
    elif x2 < w and x1 >= w:
        # Reverse order: Click 2 is Left, Click 1 is Right
        disp_x_left = x2
        disp_x_right = x1 - w
    else:
        print("Error: You must click one point in the Left image and one in the Right.")
        return None, None

    disparity = abs(disp_x_left - disp_x_right)

    if disparity == 0:
        print("Error: Zero disparity selected (points are identical). Cannot calculate depth.")
        return None, None

    Z = compute_depth_from_stereo(focal_length, baseline, disparity)
    
    print(f"\n[STEREO RESULT]")
    print(f"Disparity: {disparity} px")
    print(f"Calculated Distance (Z): {Z:.4f} cm")
    
    return Z, imgL

def measure_segments(image, focal_length, distance_Z):
    """Phase 3: Arbitrary measurements on the image."""
    print("\n" + "="*50)
    print("PHASE 3: OBJECT MEASUREMENT")
    print("Measure any shape: Width, Diameter, Polygon Edges.")
    print("="*50)
    
    measurements = []
    
    current_points = []
    display_img = image.copy()
    window_name = "Measurement Mode (Press 'q' to Quit)"
    
    print("INSTRUCTIONS:")
    print(" - Click Point A, then Point B to measure distance.")
    print(" - The real-world size will appear on screen.")
    print(" - Repeat for as many segments as needed.")
    print(" - Press 'q' when finished.")

    def mouse_handler(event, x, y, flags, param):
        nonlocal display_img, current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)
            
            # If we have a pair, measure it
            if len(current_points) % 2 == 0:
                pt1 = current_points[-2]
                pt2 = current_points[-1]
                
                # Math
                px_dist = math.dist(pt1, pt2)
                real_size = compute_real_size_from_depth(px_dist, focal_length, distance_Z)
                
                # Store
                measurements.append(real_size)
                
                # Visuals
                cv2.line(display_img, pt1, pt2, (255, 0, 0), 2)
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                draw_text(display_img, f"{real_size:.2f} cm", (mid_x, mid_y - 10))
                
                print(f"Measured Segment: {real_size:.4f} cm")

            cv2.imshow(window_name, display_img)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)
    cv2.imshow(window_name, display_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    return measurements

# --- MAIN EXECUTION ---

def main():
    try:
        # Step 1-5: Calibration
        focal_length = get_focal_length()
        
        # Step 6-10: Stereo Depth
        Z, left_img = get_stereo_depth(focal_length)
        
        if Z is None:
            print("Failed to calculate depth. Exiting.")
            return

        # Step 11: Free measurement
        results = measure_segments(left_img, focal_length, Z)
        
        # Final Summary
        print("\n" + "="*50)
        print("FINAL SUMMARY REPORT")
        print("="*50)
        print(f"Camera Focal Length : {focal_length:.2f} px")
        print(f"Object Z-Distance   : {Z:.2f} cm")
        print("-" * 30)
        print(f"Total Measurements Taken: {len(results)}")
        for i, val in enumerate(results, 1):
            print(f"Measurement #{i}: {val:.4f} cm")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()