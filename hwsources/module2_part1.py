

import cv2
import numpy as np
import os
import glob

def main():
    print("=== Template Matching Object Detector ===")
    
    # 1. Get Input Paths (1 template, many scenes)
    user_template = input("Enter path to the Template Image (e.g., template.jpg): ").strip()
    user_scene_dir = input("Enter directory containing Scene Images (e.g., ./scenes): ").strip()

    # Expand user (~) and convert to absolute paths
    template_path = os.path.abspath(os.path.expanduser(user_template))
    scene_dir = os.path.abspath(os.path.expanduser(user_scene_dir))

    # Check if paths exist
    if not os.path.exists(template_path):
        print(f"[ERROR] Template image '{template_path}' not found.")
        return

    if not os.path.isdir(scene_dir):
        print(f"[ERROR] Scene directory '{scene_dir}' not found.")
        return

    # Load Template
    template = cv2.imread(template_path, 0) # Load as grayscale
    if template is None:
        print(f"[ERROR] Could not load or decode template image '{template_path}'.")
        return
    w, h = template.shape[::-1]
    template_name = os.path.basename(template_path)

    # Get list of scene files
    scene_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        scene_files.extend(glob.glob(os.path.join(scene_dir, ext)))
    
    if not scene_files:
        print(f"[ERROR] No images found in scene directory '{scene_dir}'.")
        return

    print(f"[INFO] Found {len(scene_files)} scenes. Searching for '{template_name}'.")
    
    # Threshold for detection
    threshold = 0.8 
    
    # --- Loop through each scene ---
    for s_file in scene_files:
        # Load Scene
        img_rgb = cv2.imread(s_file)
        if img_rgb is None:
            print(f"[WARN] Could not load scene '{s_file}'. Skipping.")
            continue
        
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        print(f"\n--- Scanning Scene: {os.path.basename(s_file)} ---")
        
        # Apply Template Matching
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where correlation is above threshold
        loc = np.where(res >= threshold)
        
        detections = 0
        output_image = img_rgb.copy()

        # If matches found
        if len(loc[0]) > 0:
            # Draw rectangles for all found matches
            for pt in zip(*loc[::-1]):
                cv2.rectangle(output_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                cv2.putText(output_image, template_name, 
                           (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detections += 1
            print(f"[MATCH] Found {detections} instance(s) of '{template_name}'.")
        else:
            print(f"[INFO] No match for '{template_name}' in this scene.")

        # Show result for the current scene
        window_title = f"Detections in {os.path.basename(s_file)}"
        cv2.imshow(window_title, output_image)
        print("Press any key to continue to the next scene...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)

    print("\n[INFO] All scenes have been processed.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()