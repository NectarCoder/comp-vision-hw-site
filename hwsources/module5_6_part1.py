#!/usr/bin/env python3
"""
Real-time ArUco marker tracker (no AI/ML).

This script detects ArUco markers in a given video file and draws a green rotated box
around each detected marker. The marker ID is shown too.

Usage:
  python module5_6_part1.py input_video.mp4 [--output out.mp4]
If no video argument is supplied or the video file doesn't exist, the program prompts
you for an input path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def choose_aruco_dictionary(dict_name: Optional[str] = None):
    """Return an ArUco dictionary object by name or a sensible default.
    """
    # Map simple names to the opencv dictionary constants
    name_map = {
        None: cv2.aruco.DICT_ARUCO_ORIGINAL,
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_250": cv2.aruco.DICT_6X6_250,
        "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "april_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    key = dict_name if dict_name in name_map else None
    return cv2.aruco.getPredefinedDictionary(name_map[key])


def detect_markers(frame: np.ndarray, aruco_dict, aruco_params) -> Tuple[np.ndarray, np.ndarray]:
    """Detect ArUco markers. Returns (corners, ids) as returned by detectMarkers.
    Corners are a list of four points per marker; ids is the detected marker ID per corners.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids


def rect_from_corners(corners: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Given 4 corners of a marker, return the boxPoints and bounding center (int tuple).
    corners shape: (4, 1, 2) or (1, 4, 2) depending on detectMarkers format per marker.
    """
    # Reshape to Nx2
    pts = corners.reshape(-1, 2)
    # compute rotated rectangle
    rect = cv2.minAreaRect(pts)
    # Expand rectangle size by padding factor
    if padding != 1.0:
        (cx, cy), (w, h), angle = rect
        rect = ((cx, cy), (w * padding, h * padding), angle)
    box = cv2.boxPoints(rect).astype(int)
    center = (int(rect[0][0]), int(rect[0][1]))
    return box, center



def draw_marker_box(frame: np.ndarray, box: np.ndarray, marker_id: int) -> None:
    """Draw a rotated green box and write the marker id label.
    """
    # Draw a thick green polyline that wraps around box, close shape
    cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=3)
    # Put marker id near top-left corner
    tl = tuple(box[0])
    cv2.putText(frame, f"ID:{marker_id}", (tl[0] + 4, tl[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def open_video_capture(path: str):
    if path == "camera":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(path)


def validate_input_path(path: Optional[str]) -> Optional[str]:
    """Return a valid file path or None if invalid. 'camera' is preserved as a valid special
    token. This doesn't prompt for input (headless operation only).
    """
    if not path:
        return None
    if path == "camera":
        return path
    p = Path(path)
    if p.is_file():
        return str(p)
    return None


def ensure_writer(output_path: Optional[str], fps: float, w: int, h: int) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path.lower().endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def process_video(input_video: str, output_video: Optional[str] = None, aruco_dict_name: Optional[str] = None, show_window: bool = False, padding: float = 1.0):
    cap = open_video_capture(input_video)
    if not cap or not cap.isOpened():
        print(f"Failed to open input {input_video}")
        return

    # Prepare ArUco
    # If 'auto' requested, attempt to find a dictionary that detects a marker in the first
    # N frames. If no marker is found, fallback to the default.
    if aruco_dict_name == "auto":
        aruco_dict = None
    else:
        aruco_dict = choose_aruco_dictionary(aruco_dict_name)
    # DetectorParameters compatibility between OpenCV versions
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    else:
        aruco_params = cv2.aruco.DetectorParameters()

    # If 'auto' dictionary was requested, try to auto-detect by scanning the first frames
    if aruco_dict is None:
        # Learn the dictionary by scanning a few frames
        aruco_dict = auto_detect_dictionary(input_video, max_frames=500)
        if aruco_dict is None:
            # fallback to default
            aruco_dict = choose_aruco_dictionary(None)

    # Prepare writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = ensure_writer(output_video, fps, w, h)

    print(f"Processing {'camera' if input_video == 'camera' else input_video}: {w}x{h} @ {fps:.2f} FPS")
    if writer:
        print(f"Writing output to {output_video}")

    last_center = None
    last_box = None
    frames_no_detection = 0
    MAX_KEEP = 6  # keep showing last known position for a few frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids = detect_markers(frame, aruco_dict, aruco_params)
        if ids is not None:
            ids_flat = ids.flatten()
        else:
            ids_flat = None

        if ids is not None and len(ids) > 0:
            # draw the first detected marker and label it; if multiple exist, draw all
            for i, marker_corners in enumerate(corners):
                try:
                    marker_id = int(ids_flat[i]) if ids_flat is not None else -1
                except Exception:
                    marker_id = -1
                box, center = rect_from_corners(marker_corners, padding=padding)
                last_center = center
                last_box = box
                frames_no_detection = 0
                draw_marker_box(frame, box, marker_id)
        else:
            frames_no_detection += 1
            if frames_no_detection <= MAX_KEEP and last_box is not None:
                # show previous
                cv2.polylines(frame, [last_box], isClosed=True, color=(0, 255, 0), thickness=2)
                if last_center:
                    cv2.putText(frame, "(last)", (last_center[0] - 40, last_center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame with overlay
        if writer:
            writer.write(frame)
        # No GUI / waiting in headless mode: we simply keep processing. To abort, use process signal.

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ArUco marker tracker (no AI/ML)")
    parser.add_argument("input", nargs="?", help="Path to input video file (or 'camera' for webcam)")
    parser.add_argument("--output", "-o", help="Path to write annotated output video (optional)")
    parser.add_argument("--dict", help="Aruco dictionary name (optional) e.g. 4x4_50, 5x5_100")
    parser.add_argument("--no-window", action='store_true', help='Don\'t show GUI window (headless mode)')
    parser.add_argument("--padding", type=float, default=1.0, help='Padding factor to enlarge bounding box around the marker (default 1.0)')
    args = parser.parse_args(argv)

    input_path = args.input
    # Default to the sample file in resources if args.input is missing and that file exists
    sample_default = Path("hwsources/resources/m5_6/aruco-marker.mp4")
    if not input_path and sample_default.is_file():
        input_path = str(sample_default)

    input_valid = validate_input_path(input_path)
    if not input_valid:
        print("Invalid or missing input path. Please pass a valid video file path or 'camera' (no prompting in headless mode).")
        sys.exit(1)
    input_path = input_valid

    # Determine default output path if none provided: write to m5_6 folder with '-tracked' suffix
    if not args.output:
        in_path = Path(input_path)
        out_name = in_path.stem + "-tracked" + (".mp4" if in_path.suffix == ".mp4" else in_path.suffix)
        default_out = in_path.parent / out_name
        output_path = str(default_out)
    else:
        output_path = args.output

    process_video(input_path, output_path, args.dict if args.dict else None, show_window=False, padding=args.padding)


def auto_detect_dictionary(video_path: str, max_frames: int = 500):
    """Try a short scan through the video using several dictionaries and return
    the dictionary object that finds the first marker.
    """
    import cv2
    try:
        candidate_names = [
            "original",
            "4x4_50",
            "5x5_100",
            "6x6_250",
            "7x7_50",
            "april_36h11",
        ]
        for name in candidate_names:
            d = choose_aruco_dictionary(name)
            cap = open_video_capture(video_path)
            if not cap or not cap.isOpened():
                continue
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                params = cv2.aruco.DetectorParameters_create()
            else:
                params = cv2.aruco.DetectorParameters()
            found = False
            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                corners, ids = detect_markers(frame, d, params)
                if ids is not None and len(ids) > 0:
                    cap.release()
                    return d
            cap.release()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()
