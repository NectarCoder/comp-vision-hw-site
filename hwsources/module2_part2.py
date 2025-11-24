"""Gaussian blur and Fourier-based inverse filtering pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Gaussian & fourier transform code

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
	# Ensure kernel size is odd for symmetric filtering
	if size % 2 == 0:
		raise ValueError("Gaussian kernel size must be odd")
	# Create 2D Gaussian kernel using exponential formula
	ax = np.arange(size) - size // 2
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
	# Normalize kernel to sum to 1
	kernel /= np.sum(kernel)
	return kernel.astype(np.float32)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
	# Apply Gaussian blur using OpenCV
	return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def _kernel_fft(shape: Tuple[int, int], kernel: np.ndarray) -> np.ndarray:
	# Compute FFT of kernel with correct zero-centering
	shifted = np.fft.ifftshift(kernel)
	return np.fft.fft2(shifted, s=shape)


def inverse_filter_deblur(
	blurred: np.ndarray,
	kernel_size: int,
	sigma: float,
	balance: float = 1e-2,
	eps: float = 1e-6,
) -> np.ndarray:
	# Create Gaussian kernel that was used to blur
	kernel = gaussian_kernel(kernel_size, sigma)
	h, w = blurred.shape[:2]
	# Transform kernel to frequency domain
	kernel_fft = _kernel_fft((h, w), kernel)
	kernel_power = np.abs(kernel_fft) ** 2
	conj_kernel = np.conj(kernel_fft)
	# Transform blurred image to frequency domain
	blurred_fft = np.fft.fft2(blurred, axes=(0, 1))
	# Apply Wiener-style inverse filter in frequency domain
	restored_fft = (conj_kernel[..., None] * blurred_fft) / (kernel_power[..., None] + balance + eps)
	# Transform back to spatial domain and clip values
	restored = np.real(np.fft.ifft2(restored_fft, axes=(0, 1)))
	return np.clip(restored, 0.0, 1.0)

# Image I/O and processing code

def load_image(path: Path) -> np.ndarray:
	# Load image as float32 in [0, 1] range
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Image not found: {path}")
	return image.astype(np.float32) / 255.0


def save_image(path: Path, image: np.ndarray) -> None:
	# Persist float image array as uint8 BGR
	path.parent.mkdir(parents=True, exist_ok=True)
	image_to_save = np.clip(image * 255.0, 0, 255).astype(np.uint8)
	cv2.imwrite(str(path), image_to_save)


def derive_output_paths(
	input_path: Path,
	blur_override: Optional[str],
	restored_override: Optional[str],
) -> Tuple[Path, Path]:
	# Generate output filenames based on input filename
	stem = input_path.stem
	suffix = input_path.suffix or ".png"
	blur_path = Path(blur_override) if blur_override else input_path.with_name(f"{stem}_b{suffix}")
	restored_path = (
		Path(restored_override)
		if restored_override
		else input_path.with_name(f"{stem}_restored{suffix}")
	)
	return blur_path, restored_path


def process_image(
	input_image: Path,
	kernel_size: int,
	sigma: float,
	balance: float,
	eps: float,
	blur_output: Optional[str],
	restored_output: Optional[str],
) -> Tuple[Path, Path]:
	original = load_image(input_image)
	blurred = apply_gaussian_blur(original, kernel_size, sigma)
	restored = inverse_filter_deblur(blurred, kernel_size, sigma, balance, eps)
	blur_path, restored_path = derive_output_paths(input_image, blur_output, restored_output)
	save_image(blur_path, blurred)
	save_image(restored_path, restored)
	return blur_path, restored_path


# Main program with interactive user input

def main() -> None:
	# Prompt user for image path
	image_path_str = input("Enter the path to the image: ").strip()
	input_path = Path(image_path_str)
	
	# Use default parameters for Gaussian blur and inverse filter
	blur_path, restored_path = process_image(
		input_image=input_path,
		kernel_size=21,
		sigma=5.0,
		balance=1e-2,
		eps=1e-6,
		blur_output=None,
		restored_output=None,
	)
	print(f"Blurred image saved to {blur_path}")
	print(f"Restored image saved to {restored_path}")


if __name__ == "__main__":
	main()
