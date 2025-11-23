"""
CSC 8830 Assignment 2 - Problem 2: Convolution & Fourier Transform
Author: [Your Name]

Description:
    1. Loads an image (L).
    2. Applies Gaussian Blur to create (L_b).
    3. Recovers L from L_b using Inverse Filtering in the Frequency Domain (FFT).

    Note on Inverse Filtering:
    Direct inversion (G / H) is extremely sensitive to noise and zeros in the kernel.
    This script adds a small epsilon value to the divisor to prevent division by zero,
    a rudimentary form of regularization (Wiener Deconvolution concept).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_kernel(ksize, sigma):
    """ Creates a 2D Gaussian kernel """
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = kernel @ kernel.T
    return kernel

def main():
    print("=== Image Deblurring via Fourier Transform ===")
    
    img_path = input("Enter image path (e.g., cover.jpg): ").strip()
    img = cv2.imread(img_path, 0) # Load as grayscale for Fourier simplicity
    
    if img is None:
        print("[ERROR] Could not load image.")
        return

    rows, cols = img.shape
    
    # --- Step 1: Create Blurred Image (L_b) ---
    
    # Define Gaussian Kernel
    ksize = 21
    sigma = 5
    kernel = create_gaussian_kernel(ksize, sigma)
    
    # Apply blur spatially to get our "Observation" L_b
    img_blurred = cv2.filter2D(img, -1, kernel)


    # --- Step 2: Recover Image (Inverse Filtering) ---
    
    # 2a. Convert Image and Kernel to Frequency Domain
    # We need to pad the kernel to the size of the image for FFT multiplication/division
    kernel_padded = np.zeros_like(img, dtype=np.float32)
    
    # Place kernel at center (or corner and shift)
    # Standard approach: Place at top-left, but we must account for anchor point.
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Copy kernel to center of padded array
    kernel_padded[rows//2 - pad_h : rows//2 + pad_h + 1, 
                  cols//2 - pad_w : cols//2 + pad_w + 1] = kernel
                  
    # Shift kernel so the center is at (0,0) in freq domain
    kernel_shifted = np.fft.ifftshift(kernel_padded)
    
    # FFT of Blurred Image
    fft_image = np.fft.fft2(img_blurred)
    
    # FFT of Kernel (Optical Transfer Function)
    fft_kernel = np.fft.fft2(kernel_shifted)
    
    # 2b. Inverse Filtering (Deconvolution)
    # F_est = G / H
    # We add a small epsilon to avoid division by zero
    epsilon = 1e-3
    
    # Perform division in frequency domain
    fft_recovered = fft_image / (fft_kernel + epsilon)
    
    # 2c. Inverse FFT to get back to Spatial Domain
    img_recovered = np.fft.ifft2(fft_recovered)
    img_recovered = np.abs(img_recovered) # Take magnitude
    
    # Normalize for display
    img_recovered = cv2.normalize(img_recovered, None, 0, 255, cv2.NORM_MINMAX)
    img_recovered = np.uint8(img_recovered)

    # --- Step 3: Visualization ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original (L)")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Blurred (L_b)\nSigma={sigma}")
    plt.imshow(img_blurred, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Recovered via FFT")
    plt.imshow(img_recovered, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()