import numpy as np
import cv2

def gaussian_blur(drr: np.ndarray, mask: np.ndarray, sigma: float) -> tuple:
    """
    Applies Gaussian blur to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        sigma: Standard deviation for Gaussian blur.
    Returns:
        Tuple of blurred DRR and unchanged mask (NumPy arrays).
    """
    ksize = int(2 * round(3 * sigma) + 1)  # Kernel size based on sigma
    blurred_drr = cv2.GaussianBlur(drr, (ksize, ksize), sigma)
    return blurred_drr, mask

def gaussian_sharpening(drr: np.ndarray, mask: np.ndarray, amount: float) -> tuple:
    """
    Enhances edge details in the DRR image using Gaussian sharpening.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        amount: Intensity of sharpening (higher value = stronger sharpening).
    Returns:
        Tuple of sharpened DRR and unchanged mask (NumPy arrays).
    """
    blurred_drr = cv2.GaussianBlur(drr, (0, 0), 3)  # Apply a soft Gaussian blur
    sharpened_drr = cv2.addWeighted(drr, 1 + amount, blurred_drr, -amount, 0)  # Sharpen using weighted subtraction
    return sharpened_drr, mask
