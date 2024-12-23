import numpy as np
import cv2

def apply_gamma_adjustment(drr: np.ndarray, mask: np.ndarray, gamma: float) -> tuple:
    """
    Adjusts the gamma of the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        gamma: Gamma value for adjustment (e.g., >1 = brighter, <1 = darker).
    Returns:
        Tuple of gamma-adjusted DRR and unchanged mask (NumPy arrays).
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    adjusted_drr = cv2.LUT(drr, table)
    return adjusted_drr, mask

def adjust_brightness(drr: np.ndarray, mask: np.ndarray, beta: float) -> tuple:
    """
    Adjusts the brightness of the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        beta: Brightness adjustment factor (-100 to 100; negative = darker, positive = brighter).
    Returns:
        Tuple of brightness-adjusted DRR and unchanged mask (NumPy arrays).
    """
    adjusted_drr = cv2.convertScaleAbs(drr, alpha=1.0, beta=beta)
    return adjusted_drr, mask

def adjust_contrast(drr: np.ndarray, mask: np.ndarray, alpha: float) -> tuple:
    """
    Adjusts the contrast of the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        alpha: Contrast adjustment factor (e.g., 1.0 = no change, <1 = lower contrast, >1 = higher contrast).
    Returns:
        Tuple of contrast-adjusted DRR and unchanged mask (NumPy arrays).
    """
    adjusted_drr = cv2.convertScaleAbs(drr, alpha=alpha, beta=0)
    return adjusted_drr, mask

def adjust_contrast_with_curve(drr: np.ndarray, mask: np.ndarray, control_points: list) -> tuple:
    """
    Adjusts the contrast of a DRR image based on intensity mapping defined by control points.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        control_points: List of tuples [(input1, output1), (input2, output2), ...].
            Define the intensity mapping curve, where input is in [0, 255] and output is normalized [0, 1].
    Returns:
        Tuple of contrast-adjusted DRR and unchanged mask (NumPy arrays).
    """
    # Create a lookup table for intensity mapping
    x = np.array([pt[0] for pt in control_points])  # Input intensities
    y = np.array([pt[1] for pt in control_points])  # Mapped intensities (scaled to 0-255)
    y_scaled = np.clip(y * 255, 0, 255).astype(np.uint8)  # Scale y to [0, 255]

    # Interpolate the intensity mapping curve
    mapping = np.interp(np.arange(256), x, y_scaled)

    # Apply the mapping to the DRR image
    adjusted_drr = cv2.LUT(drr, mapping.astype(np.uint8))

    return adjusted_drr, mask