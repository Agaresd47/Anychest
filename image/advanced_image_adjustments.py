import numpy as np
import cv2
import math

def bias_field_simulation(drr: np.ndarray, mask: np.ndarray, max_bias: float) -> tuple:
    """
    Simulates a bias field (brightness gradient) on the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        max_bias: Maximum intensity change for the bias field.
    Returns:
        Tuple of bias-field-simulated DRR and unchanged mask (NumPy arrays).
    """
    h, w = drr.shape
    gradient = np.linspace(1 - max_bias, 1 + max_bias, w)
    bias_field = np.tile(gradient, (h, 1)).astype(np.float32)
    biased_drr = np.clip(drr * bias_field, 0, 255).astype(np.uint8)
    return biased_drr, mask

def histogram_equalization(drr: np.ndarray, mask: np.ndarray, process: bool) -> tuple:
    """
    Applies histogram equalization to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
    Returns:
        Tuple of histogram-equalized DRR and unchanged mask (NumPy arrays).
    """
    if not process:
        return 
    equalized_drr = cv2.equalizeHist(drr)
    return equalized_drr, mask

def boundary_smoothing(drr: np.ndarray, mask: np.ndarray, kernel_size: int) -> tuple:
    """
    Smooths the boundaries in the DRR image using a Gaussian blur.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        kernel_size: Size of the Gaussian kernel for smoothing.
    Returns:
        Tuple of smoothed DRR and unchanged mask (NumPy arrays).
    """
    kernel_size = math.ceil(kernel_size)
    
    if kernel_size % 2 == 0:  # Convert even to odd
        kernel_size += 1
        
    smoothed_drr = cv2.GaussianBlur(drr, (kernel_size, kernel_size), 0)
    return smoothed_drr, mask
