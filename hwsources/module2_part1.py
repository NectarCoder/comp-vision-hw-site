# TODO DESCRIPTION

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_THRESHOLD = 0.9
CORRELATION_METHOD = cv2.TM_CCOEFF_NORMED
IOU_SUPPRESSION = 0.3
NORMALIZE_EPS = 1e-6


@dataclass
class MatchRegion:
	score: float
	top_left: Tuple[int, int]
	bottom_right: Tuple[int, int]


@dataclass
class DetectionResult:
	matches: List[MatchRegion]
	annotated_image: Optional[Path]


def load_image(path: Path, grayscale: bool) -> np.ndarray:
	flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
	image = cv2.imread(str(path), flag)
	if image is None:
		raise FileNotFoundError(f"Unable to read image: {path}")
	return image


def match_template(template: np.ndarray, search: np.ndarray) -> np.ndarray:
	if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
		raise ValueError("Template must be smaller than the search image")
	return cv2.matchTemplate(search, template, CORRELATION_METHOD)


def annotate_detections(image: np.ndarray, matches: List[MatchRegion]) -> np.ndarray:
	output = image.copy()
	for match in matches:
		cv2.rectangle(output, match.top_left, match.bottom_right, (0, 255, 0), 2)
	return output


def detect_object(template_path: Path, search_path: Path, threshold: float) -> DetectionResult:
	template_gray = enhance_for_matching(load_image(template_path, grayscale=True))
	search_gray = enhance_for_matching(load_image(search_path, grayscale=True))
	search_color = load_image(search_path, grayscale=False)
	response = match_template(template_gray, search_gray)
	matches = find_match_regions(response, template_gray.shape[::-1], threshold)
	annotated_path: Optional[Path] = None
	if matches:
		annotated = annotate_detections(search_color, matches)
		suffix = search_path.suffix or ".png"
		annotated_path = search_path.with_name(f"{search_path.stem}_detections{suffix}")
		annotated_path.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(annotated_path), annotated)
	return DetectionResult(matches, annotated_path)


def prompt_path(prompt_text: str) -> Path:
	return Path(input(prompt_text).strip())


def find_match_regions(
	response: np.ndarray,
	template_size: Tuple[int, int],
	threshold: float,
) -> List[MatchRegion]:
	if response.size == 0:
		return []
	kernel_size = _nms_kernel_size(template_size)
	kernel = np.ones(kernel_size, dtype=np.uint8)
	max_map = cv2.dilate(response, kernel)
	mask = (response >= threshold) & (response == max_map)
	locations = np.argwhere(mask)
	matches: List[MatchRegion] = []
	for (y, x) in locations:
		score = float(response[y, x])
		top_left = (int(x), int(y))
		bottom_right = (top_left[0] + template_size[0], top_left[1] + template_size[1])
		matches.append(MatchRegion(score, top_left, bottom_right))
	matches.sort(key=lambda m: m.score, reverse=True)
	return _suppress_overlaps(matches, IOU_SUPPRESSION)


def enhance_for_matching(image: np.ndarray) -> np.ndarray:
	"""Improve grayscale contrast and normalize distribution for template matching."""
	if image.dtype != np.uint8:
		image = image.astype(np.uint8)
	equalized = cv2.equalizeHist(image)
	float_img = equalized.astype(np.float32) / 255.0
	mean = float(np.mean(float_img))
	std = float(np.std(float_img)) + NORMALIZE_EPS
	standardized = (float_img - mean) / std
	return standardized


def _nms_kernel_size(template_size: Tuple[int, int]) -> Tuple[int, int]:
	width, height = template_size
	w = max(1, width // 4)
	h = max(1, height // 4)
	if w % 2 == 0:
		w += 1
	if h % 2 == 0:
		h += 1
	return w, h


def _suppress_overlaps(matches: List[MatchRegion], iou_threshold: float) -> List[MatchRegion]:
	kept: List[MatchRegion] = []
	for candidate in matches:
		if all(_iou(candidate, existing) < iou_threshold for existing in kept):
			kept.append(candidate)
	return kept


def _iou(a: MatchRegion, b: MatchRegion) -> float:
	ax1, ay1 = a.top_left
	ax2, ay2 = a.bottom_right
	bx1, by1 = b.top_left
	bx2, by2 = b.bottom_right
	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)
	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h
	a_area = (ax2 - ax1) * (ay2 - ay1)
	b_area = (bx2 - bx1) * (by2 - by1)
	union = a_area + b_area - inter_area
	if union <= 0:
		return 0.0
	return inter_area / union


def prompt_threshold() -> float:
	raw_value = input(
		f"Enter detection threshold between 0 and 1 [{DEFAULT_THRESHOLD}]: "
	).strip()
	if not raw_value:
		return DEFAULT_THRESHOLD
	try:
		threshold = float(raw_value)
	except ValueError:
		print("Invalid threshold provided. Using default value.")
		return DEFAULT_THRESHOLD
	if not 0.0 < threshold <= 1.0:
		print("Threshold must be within (0, 1]. Using default value.")
		return DEFAULT_THRESHOLD
	return threshold


def main() -> None:
	print("Template Matching using normalized cross-correlation")
	print("Provide a search image that contains the full scene and a smaller template of just the object.")
	search_path = prompt_path("Enter the path to the search image: ")
	template_path = prompt_path("Enter the path to the template image: ")
	threshold = prompt_threshold()
	try:
		result = detect_object(template_path, search_path, threshold)
	except FileNotFoundError as exc:
		print(exc)
		return
	except ValueError as exc:
		print(f"Error: {exc}")
		return
	if result.matches:
		top_match = result.matches[0]
		print(
			f"Detected {len(result.matches)} match(es) with top score {top_match.score:.3f}"
		)
		for idx, match in enumerate(result.matches, start=1):
			print(
				f"  #{idx}: score={match.score:.3f}, top-left={match.top_left}, bottom-right={match.bottom_right}"
			)
		if result.annotated_image:
			print(f"Annotated detections saved to: {result.annotated_image}")
	else:
		print("No matches exceeded the provided threshold.")



if __name__ == "__main__":
	main()

