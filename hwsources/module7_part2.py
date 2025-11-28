from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import mediapipe as mp


def _call_with_fallback(module, attr: str, fallback):
    fn = getattr(module, attr, None)
    if callable(fn):
        return fn()
    return fallback


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose and hand landmark extraction with optional visualization.",
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the input video file to analyze.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a real-time window with the detected landmarks (default: enabled).",
    )
    return parser.parse_args(argv)


def _build_fieldnames() -> List[str]:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    fieldnames: List[str] = ["frame_index", "time_sec"]

    for landmark in mp_pose.PoseLandmark:
        name = landmark.name.lower()
        fieldnames.extend(
            [
                f"pose_{name}_x",
                f"pose_{name}_y",
                f"pose_{name}_z",
                f"pose_{name}_visibility",
            ]
        )

    for side in ("left", "right"):
        for landmark in mp_hands.HandLandmark:
            name = landmark.name.lower()
            fieldnames.extend(
                [
                    f"{side}_hand_{name}_x",
                    f"{side}_hand_{name}_y",
                    f"{side}_hand_{name}_z",
                ]
            )

    return fieldnames


def _derive_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_pose_hand.csv")


def _initialize_frame_record(fieldnames: List[str]) -> Dict[str, float]:
    return {field: None for field in fieldnames}


def _run_estimation(video_path: Path, display: bool) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fieldnames = _build_fieldnames()
    records: List[Dict[str, float]] = []

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose_landmark_spec = _call_with_fallback(
        mp_drawing_styles,
        "get_default_pose_landmarks_style",
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    )
    pose_connection_spec = _call_with_fallback(
        mp_drawing_styles,
        "get_default_pose_connection_style",
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
    )
    hand_landmark_spec = _call_with_fallback(
        mp_drawing_styles,
        "get_default_hand_landmarks_style",
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
    )
    hand_connection_spec = _call_with_fallback(
        mp_drawing_styles,
        "get_default_hand_connections_style",
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
    )

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    ) as hands:
        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                pose_results = pose.process(frame_rgb)
                hands_results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                record = _initialize_frame_record(fieldnames)
                record["frame_index"] = frame_index
                record["time_sec"] = frame_index / fps if fps > 0 else None

                if pose_results.pose_landmarks:
                    for landmark_enum, landmark in zip(
                        mp_pose.PoseLandmark,
                        pose_results.pose_landmarks.landmark,
                    ):
                        name = landmark_enum.name.lower()
                        record[f"pose_{name}_x"] = landmark.x
                        record[f"pose_{name}_y"] = landmark.y
                        record[f"pose_{name}_z"] = landmark.z
                        record[f"pose_{name}_visibility"] = landmark.visibility

                handedness_map: Dict[int, str] = {}
                if hands_results.multi_handedness:
                    for idx, hand_info in enumerate(hands_results.multi_handedness):
                        handedness_map[idx] = hand_info.classification[0].label.lower()

                if hands_results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                        side = handedness_map.get(idx, f"hand_{idx}")
                        if side not in ("left", "right"):
                            side = "left" if "left" not in handedness_map.values() else "right"
                        for landmark_enum, landmark in zip(
                            mp_hands.HandLandmark,
                            hand_landmarks.landmark,
                        ):
                            name = landmark_enum.name.lower()
                            record[f"{side}_hand_{name}_x"] = landmark.x
                            record[f"{side}_hand_{name}_y"] = landmark.y
                            record[f"{side}_hand_{name}_z"] = landmark.z

                records.append(record)

                if display:
                    annotated = frame.copy()
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=pose_landmark_spec,
                            connection_drawing_spec=pose_connection_spec,
                        )
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=hand_landmark_spec,
                                connection_drawing_spec=hand_connection_spec,
                            )

                    cv2.putText(
                        annotated,
                        "Press 'q' to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Pose and Hand Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    output_path = _derive_output_path(video_path)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return output_path


def main(argv: Iterable[str]) -> None:
    args = _parse_args(argv)
    try:
        output_path = _run_estimation(args.video_path, args.display)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    print(f"Pose and hand landmarks saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
