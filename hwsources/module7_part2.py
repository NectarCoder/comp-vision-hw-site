"""
CSC 8830 Assignment 7 - Problem 3: Real-time Pose & Hand Tracking
Author: [Your Name]

Description:
    1. Captures webcam feed.
    2. Uses MediaPipe to detect Body Pose and Hand Landmarks.
    3. Visualizes the skeleton on screen.
    4. Saves raw landmark data (X, Y, Z, Visibility) to a CSV file in real-time.
    
Dependencies:
    pip install mediapipe opencv-python
"""

import cv2
import mediapipe as mp
import csv
import time
import os

def main():
    # --- Setup MediaPipe ---
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # Initialize Detectors
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # --- Setup CSV Logging ---
    csv_filename = "pose_hand_data.csv"
    file_exists = os.path.isfile(csv_filename)
    
    csv_file = open(csv_filename, mode='a', newline='')
    writer = csv.writer(csv_file)
    
    # Write Header if new file
    if not file_exists:
        # Structure: Timestamp, Type (Pose/LeftHand/RightHand), Landmark_ID, X, Y, Z, Visibility
        writer.writerow(["Timestamp", "Type", "ID", "X", "Y", "Z", "Visibility"])

    print(f"[INFO] Logging data to {csv_filename}...")
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR to RGB for MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        
        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        timestamp = time.time()

        # --- 1. Process Pose ---
        if pose_results.pose_landmarks:
            # Draw
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Log Data
            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                # x, y are normalized [0.0, 1.0]
                writer.writerow([timestamp, "Pose", idx, lm.x, lm.y, lm.z, lm.visibility])

        # --- 2. Process Hands ---
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Draw
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                label = handedness.classification[0].label # "Left" or "Right"
                
                # Log Data
                for idx, lm in enumerate(hand_landmarks.landmark):
                    # Hands don't usually return visibility, defaulting to 1.0
                    writer.writerow([timestamp, f"Hand_{label}", idx, lm.x, lm.y, lm.z, 1.0])

        # --- Display ---
        cv2.imshow('Pose & Hand Tracking (Assignment 7)', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("[INFO] Tracking stopped. Data saved.")

if __name__ == "__main__":
    main()