import numpy as np
import cv2

def spatial_cropping(drr: np.ndarray, mask: np.ndarray, crop_size: tuple) -> tuple:
    """
    Crops the DRR image and mask to a specified size.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        crop_size: Tuple (height, width) specifying the crop size.
    Returns:
        Tuple of cropped DRR and mask (NumPy arrays).
    """
    h, w = drr.shape[:2]
    ch, cw = crop_size

    # Randomly select the top-left corner for cropping
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)

    cropped_drr = drr[top:top + ch, left:left + cw]
    cropped_mask = mask[top:top + ch, left:left + cw]

    return cropped_drr, cropped_mask

def rotate_image(drr: np.ndarray, mask: np.ndarray, angle: float) -> tuple:
    """
    Rotates the DRR image and mask by a specified angle.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        angle: The rotation angle in degrees.
    Returns:
        Tuple of rotated DRR and mask (NumPy arrays).
    """
    h, w = drr.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_drr = cv2.warpAffine(drr, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    return rotated_drr, rotated_mask

def axis_aligned_flip(drr: np.ndarray, mask: np.ndarray, direction: str) -> tuple:
    """
    Flips the DRR image and mask along the specified axis.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        direction: 'horizontal' or 'vertical'.
    Returns:
        Tuple of flipped DRR and mask (NumPy arrays).
    """
    if direction == 'horizontal':
        flipped_drr = cv2.flip(drr, 1)
        flipped_mask = cv2.flip(mask, 1)
    elif direction == 'vertical':
        flipped_drr = cv2.flip(drr, 0)
        flipped_mask = cv2.flip(mask, 0)
    else:
        raise ValueError("Invalid direction for flip. Use 'horizontal' or 'vertical'.")
    return flipped_drr, flipped_mask

def perspective_transform(drr: np.ndarray, mask: np.ndarray, points: list) -> tuple:
    """
    Applies a perspective transformation to the DRR image and mask.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        points: List of 4 points defining the transformation.
    Returns:
        Tuple of transformed DRR and mask (NumPy arrays).
    """
    h, w = drr.shape[:2]
    src = np.float32(points)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    transformed_drr = cv2.warpPerspective(drr, matrix, (w, h), flags=cv2.INTER_LINEAR)
    transformed_mask = cv2.warpPerspective(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return transformed_drr, transformed_mask

def zoom_and_crop(drr: np.ndarray, mask: np.ndarray, scale: float) -> tuple:
    """
    Applies a zoom effect followed by cropping to the DRR image and mask.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        scale: Zoom scale factor (>1 for zoom-in, <1 for zoom-out).
    Returns:
        Tuple of zoomed and cropped DRR and mask (NumPy arrays).
    """
    h, w = drr.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    zoomed_drr = cv2.resize(drr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    zoomed_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Center crop back to original size
    top = (nh - h) // 2
    left = (nw - w) // 2
    cropped_drr = zoomed_drr[top:top + h, left:left + w]
    cropped_mask = zoomed_mask[top:top + h, left:left + w]

    return cropped_drr, cropped_mask

def random_occlusions(drr: np.ndarray, mask: np.ndarray, num_occlusions: int, size: tuple) -> tuple:
    """
    Adds random occlusions to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        num_occlusions: Number of occlusions to apply.
        size: Tuple (height, width) specifying the occlusion size.
    Returns:
        Tuple of occluded DRR and unchanged mask (NumPy arrays).
    """
    occluded_drr = drr.copy()
    for _ in range(num_occlusions):
        h, w = drr.shape[:2]
        top = np.random.randint(0, h - size[0])
        left = np.random.randint(0, w - size[1])
        occluded_drr[top:top + size[0], left:left + size[1]] = 0
    return occluded_drr, mask
