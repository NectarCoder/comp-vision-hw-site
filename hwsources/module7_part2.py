from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _run_estimation(video_path: Path, display: bool, annotated_output_path: Optional[Path] = None) -> Dict[str, object]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps_safe = fps if fps and fps > 0 else 30.0
    fieldnames = _build_fieldnames()
    records: List[Dict[str, float]] = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    annotated_path = Path(annotated_output_path) if annotated_output_path else None
    video_writer = None
    if annotated_path:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

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

                if frame_width <= 0 or frame_height <= 0:
                    frame_height, frame_width = frame.shape[:2]

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

                should_draw = display or annotated_path is not None
                annotated_frame = None
                if should_draw:
                    annotated_frame = frame.copy()
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=pose_landmark_spec,
                            connection_drawing_spec=pose_connection_spec,
                        )
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=hand_landmark_spec,
                                connection_drawing_spec=hand_connection_spec,
                            )

                    if annotated_path and video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(annotated_path),
                            fourcc,
                            fps_safe,
                            (frame_width or frame.shape[1], frame_height or frame.shape[0]),
                        )
                        if not video_writer.isOpened():
                            video_writer.release()
                            video_writer = None

                    if video_writer is not None:
                        video_writer.write(annotated_frame)

                    if display:
                        cv2.putText(
                            annotated_frame,
                            "Press 'q' to quit",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Pose and Hand Tracking", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_index += 1
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

    output_path = _derive_output_path(video_path)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    duration = frame_index / fps if fps > 0 else None
    result = {
        "csv_path": output_path,
        "record_count": len(records),
        "frame_count": frame_index,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "duration_seconds": duration,
        "annotated_video_path": annotated_path if annotated_path and annotated_path.exists() else None,
    }
    return result


def main(argv: Iterable[str]) -> None:
    args = _parse_args(argv)
    try:
        result = _run_estimation(args.video_path, args.display)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    output_path = result.get("csv_path")
    if output_path:
        print(f"Pose and hand landmarks saved to {output_path}")
    annotated_path = result.get("annotated_video_path")
    if annotated_path:
        print(f"Annotated video exported to {annotated_path}")


def run_pose_and_hand_tracking(
    video_path: Path,
    *,
    display: bool = False,
    annotated_output_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Programmatic helper so other modules (e.g., Flask app) can reuse this pipeline."""

    return _run_estimation(video_path, display=display, annotated_output_path=annotated_output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
