"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 1 Assignment - Object Dimension Calculator

The purpose of the script is for calculating the
dimensions of an object by using the perspective
projection equations (Pinhole camera model).

Equation used is W/D = w/f --> W = (w*D)/f, where
 - w is the object's width in terms of pixels
 - f is the camera's focal length in terms of pixels
 - W is the object's width in real life (cm)
 - D is the distance between camera and object (cm)

Process:
 1. User provides REFERENCE image path
 2. User selects two points on the reference image to mark object width
 3. User inputs real-world width and distance for the reference object
 4. Program calculates and stores focal length
 5. User provides TEST image path (must be from same camera)
 6. User selects two points on the test image to mark object width
 7. User inputs distance to the test object
 8. Program calculates and displays real-world width of test object

How to run the program:
 - Run this script: python module1.py
 - Follow the sequential prompts

The dependencies are opencv-python
"""

import cv2
import math
import os


def calculate_focal_length(pixel_width, real_width, distance):
    """Compute focal length (pixels) from pixel width, real width, and distance."""
    if real_width == 0:
        raise ValueError("Real width must be non-zero")
    return (pixel_width * distance) / real_width


def calculate_real_dimension(pixel_width, focal_length, distance):
    """Compute real-world dimension from pixel width, focal length, and distance."""
    if focal_length == 0:
        raise ValueError("Focal length must be non-zero")
    return (pixel_width * distance) / focal_length


def load_image(image_path, max_height=800):
    """Load an image, optionally resizing by height for consistent display."""
    candidate = os.path.expanduser(image_path)
    if not os.path.exists(candidate):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, image_path)
        if not os.path.exists(candidate):
            return None

    image = cv2.imread(candidate)
    if image is None:
        return None

    height, width = image.shape[:2]
    if height <= max_height:
        return image

    scale = max_height / height
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size)


def select_two_points_on_image(image, window_name="Select Points"):
    """Display `image` and let user pick two points. Returns (point1, point2) or (None, None)."""
    points = []
    display_image = image.copy()

    def mouse_callback(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) == 2:
            return

        points.append((x, y))
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)
            dist_px = math.dist(points[0], points[1])
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            cv2.putText(
                display_image,
                f"{dist_px:.1f}px",
                (mid_x, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, display_image)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_image)

    print("\n[INFO] Click TWO points on the image to mark the object width.")
    print("[INFO] Press 'q' to cancel.")

    while len(points) < 2:
        if cv2.waitKey(50) & 0xFF == ord("q"):
            cv2.destroyWindow(window_name)
            return None, None

    cv2.waitKey(300)
    cv2.destroyWindow(window_name)
    return tuple(points) if len(points) == 2 else (None, None)


def prompt_float(prompt_text):
    """Read a float from stdin. Returns None if parsing fails."""
    try:
        return float(input(prompt_text).strip())
    except ValueError:
        print("[ERROR] Invalid numeric input.")
        return None


def prompt_image(label, prompt_text):
    """Prompt for an image path and return the loaded image or None."""
    image_path = input(prompt_text).strip()
    image = load_image(image_path)
    if image is None:
        print(f"[ERROR] Could not load {label.lower()} image: {image_path}")
    else:
        print(f"[INFO] {label} image loaded successfully.")
    return image


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_intro():
    print("=" * 70)
    print("          Object Dimension Calculator          ")
    print("=" * 70)
    print("\n[IMPORTANT] Both the REFERENCE and TEST images MUST be taken by the")
    print("            SAME CAMERA for accurate results!\n")


def calibrate_camera():
    print_section("STEP 1: REFERENCE IMAGE (Calibration)")
    ref_image = prompt_image("REFERENCE", "\nEnter the path to the REFERENCE image: ")
    if ref_image is None:
        return None

    point_a, point_b = select_two_points_on_image(
        ref_image, "REFERENCE Image - Select 2 Points"
    )
    if point_a is None or point_b is None:
        print("[ERROR] Point selection cancelled.")
        return None

    ref_pixel_width = math.dist(point_a, point_b)
    print(f"\n[INFO] Reference pixel width: {ref_pixel_width:.2f} pixels")

    ref_real_width = prompt_float("Enter the REAL-WORLD WIDTH of the object (cm): ")
    if ref_real_width is None:
        return None

    ref_distance = prompt_float("Enter the DISTANCE from camera to object (cm): ")
    if ref_distance is None:
        return None

    try:
        focal_length = calculate_focal_length(
            ref_pixel_width, ref_real_width, ref_distance
        )
    except ValueError as err:
        print(f"[ERROR] {err}")
        return None

    print("\n" + "-" * 70)
    print(f"[RESULT] CALCULATED FOCAL LENGTH: {focal_length:.2f} pixels")
    print("-" * 70)
    print("[INFO] Focal length stored for test image measurement.\n")
    return focal_length


def measure_test_object(focal_length):
    print_section("STEP 2: TEST IMAGE (Measurement)")
    test_image = prompt_image("TEST", "\nEnter the path to the TEST image: ")
    if test_image is None:
        return

    point_a, point_b = select_two_points_on_image(
        test_image, "TEST Image - Select 2 Points"
    )
    if point_a is None or point_b is None:
        print("[ERROR] Point selection cancelled.")
        return

    test_pixel_width = math.dist(point_a, point_b)
    print(f"\n[INFO] Test pixel width: {test_pixel_width:.2f} pixels")

    test_distance = prompt_float(
        "Enter the DISTANCE from camera to the test object (cm): "
    )
    if test_distance is None:
        return

    try:
        test_real_width = calculate_real_dimension(
            test_pixel_width, focal_length, test_distance
        )
    except ValueError as err:
        print(f"[ERROR] {err}")
        return

    print_section("FINAL RESULT")
    print(f"Focal Length Used:        {focal_length:.2f} pixels")
    print(f"Test Pixel Width:         {test_pixel_width:.2f} pixels")
    print(f"Test Object Distance:     {test_distance:.2f} cm")
    print(f"\n>>> CALCULATED REAL-WORLD WIDTH: {test_real_width:.4f} cm <<<")
    print("=" * 70)
    print("\n[INFO] Calculation complete. Program ending.")


def main():
    """Run CLI to compute object dimensions using the pinhole camera model."""
    show_intro()
    focal_length = calibrate_camera()
    if focal_length is None:
        return
    measure_test_object(focal_length)


if __name__ == "__main__":
    main()
