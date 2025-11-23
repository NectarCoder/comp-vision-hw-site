"""
CSC 8830 Assignment 5-6: Real-Time Object Tracking Suite
Author: [Your Name]

Description:
    Implements the requirements for Assignment 5-6 Problem 2 & 1(a).
    
    Modes:
    1. 'marker': Real-time tracking using ArUco markers.
    2. 'markerless': Real-time tracking using OpenCV CSRT (User selects object).
    3. 'sam2': overlays pre-computed segmentation masks (from NPZ file).
    4. 'flow': Computes dense optical flow between frames (Problem 1a).

Usage:
    python assignment5_tracking_suite.py
    
Controls:
    'q' - Quit
    'm' - Switch to Marker Mode
    't' - Switch to Marker-less Tracking Mode (Select ROI)
    's' - Switch to SAM2 Overlay Mode
    'f' - Capture Motion Estimate (Optical Flow) snapshot
"""

import cv2
import numpy as np
import os

# --- Global Constants ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
SAM_DATA_FILE = "sam2_masks.npz"

class TrackingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # Use 0 for webcam
        self.mode = "marker" # Default mode
        self.tracker = None
        self.tracking_initialized = False
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # SAM2 Data container
        self.sam_masks = {}
        self.frame_count = 0
        self.load_sam_data()

    def load_sam_data(self):
        """ Loads offline segmentation masks for Mode 3 """
        if os.path.exists(SAM_DATA_FILE):
            print(f"[INFO] Loading SAM2 data from {SAM_DATA_FILE}...")
            data = np.load(SAM_DATA_FILE)
            # Convert npz keys to integer frames
            self.sam_masks = {int(k.replace('frame_', '')): v for k, v in data.items()}
            print(f"[INFO] Loaded {len(self.sam_masks)} frames of masks.")
        else:
            print(f"[WARNING] {SAM_DATA_FILE} not found. SAM2 mode will show empty overlay.")
            print("          Run 'generate_dummy_sam_data.py' to create test data.")

    def process_marker_mode(self, frame):
        """ Mode (i): Uses ArUco markers """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        vis = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            
            # Draw tracking info
            for c in corners:
                cx = int(np.mean(c[0][:, 0]))
                cy = int(np.mean(c[0][:, 1]))
                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis, f"Marker Pos: {cx},{cy}", (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No Marker Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return vis

    def process_markerless_mode(self, frame):
        """ Mode (ii): Uses CSRT Tracker """
        vis = frame.copy()
        
        if not self.tracking_initialized:
            cv2.putText(vis, "Press SPACE to select Object to Track", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return vis
            
        success, box = self.tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "Tracker: Active", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Tracker: Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return vis

    def process_sam2_mode(self, frame):
        """ Mode (iii): Uses Offline SAM2 Segmentation NPZ """
        vis = frame.copy()
        
        # Simulate frame-sync (using modulo to loop the dummy data)
        current_idx = self.frame_count % max(1, len(self.sam_masks))
        
        if current_idx in self.sam_masks:
            mask = self.sam_masks[current_idx]
            # Resize mask if it doesn't match current frame (e.g. if webcam res changed)
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
            
            # Apply colored overlay
            color_mask = np.zeros_like(frame)
            color_mask[:, :] = [0, 0, 255] # Red overlay
            
            # Blend
            # mask is boolean or 0/1
            mask_indices = mask > 0
            vis[mask_indices] = cv2.addWeighted(vis[mask_indices], 0.6, color_mask[mask_indices], 0.4, 0)
            
            cv2.putText(vis, f"SAM2 Overlay: Frame {current_idx}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(vis, "No SAM2 Data", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
        return vis

    def compute_optical_flow(self):
        """ Problem 1(a): Compute Motion Estimate between 2 frames """
        print("[INFO] Capturing Frame 1...")
        ret, frame1 = self.cap.read()
        if not ret: return
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        print("[INFO] Waiting 200ms for motion...")
        cv2.waitKey(200) # Wait a bit for motion to happen
        
        print("[INFO] Capturing Frame 2...")
        ret, frame2 = self.cap.read()
        if not ret: return
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate Dense Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Visualize
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imshow("Motion Estimate (Prob 1a)", rgb)
        print("[INFO] Motion estimate displayed.")

    def run(self):
        print("=== Object Tracker Started ===")
        print("Modes: [m]arker, [t]racker (markerless), [s]am2, [f]low snapshot")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.frame_count += 1
            
            # Processing based on mode
            if self.mode == 'marker':
                output = self.process_marker_mode(frame)
            elif self.mode == 'markerless':
                output = self.process_markerless_mode(frame)
            elif self.mode == 'sam2':
                output = self.process_sam2_mode(frame)
            else:
                output = frame
                
            # UI Text
            cv2.putText(output, f"Mode: {self.mode.upper()} (Press m/t/s/f)", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Tracking Suite", output)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Controls
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.mode = 'marker'
                print("[MODE] Switched to Marker Tracking")
            elif key == ord('t'):
                self.mode = 'markerless'
                print("[MODE] Switched to Markerless Tracking")
                # Trigger selection
                bbox = cv2.selectROI("Tracking Suite", frame, fromCenter=False, showCrosshair=True)
                self.tracker = cv2.TrackerCSRT_create() # CSRT is robust
                self.tracker.init(frame, bbox)
                self.tracking_initialized = True
            elif key == ord('s'):
                self.mode = 'sam2'
                print("[MODE] Switched to SAM2 Overlay")
            elif key == ord('f'):
                self.compute_optical_flow()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = TrackingApp()
    app.run()