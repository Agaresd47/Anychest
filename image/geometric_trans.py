import numpy as np
import cv2
import math

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
    
    # Get the original size dynamically
    target_size = (h, w)
    
    # Resize cropped images to the target size
    resized_drr = cv2.resize(cropped_drr, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return resized_drr, resized_mask

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
    
    
        # Get the original size dynamically
    target_size = (h, w)
    
    # Resize cropped images to the target size
    resized_drr = cv2.resize(transformed_drr, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(transformed_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    return resized_drr, resized_mask

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
    
            # Get the original size dynamically
    target_size = (h, w)
    
    # Resize cropped images to the target size
    resized_drr = cv2.resize(cropped_drr, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return resized_drr, resized_mask

def random_occlusions(drr: np.ndarray, mask: np.ndarray, num_occlusions: int, size: tuple):
    """
    Apply random occlusions to the DRR image. Each occlusion gets a unique ID (0 to num_occlusions-1).
    The mask remains unchanged.

    Args:
        drr: The DRR image (NumPy array).
        mask: The corresponding mask (NumPy array) - remains unchanged.
        num_occlusions: The number of occlusions to apply.
        size: A tuple specifying the size of the occlusion (height, width).

    Returns:
        Tuple: The occluded DRR image and the unchanged mask.
    """
    num_occlusions = math.ceil(num_occlusions)
    
    # Create a copy of the DRR to apply occlusions to
    occluded_drr = drr.copy()

    # Generate unique occlusion IDs (uniformly distributed) for each occlusion
    occlusion_ids = np.random.randint(0, num_occlusions, num_occlusions)  # Uniform distribution of occlusion IDs

    for i in range(num_occlusions):
        # Get image dimensions
        h, w = drr.shape[:2]
        
        # Randomly select the top-left corner for the occlusion
        top = np.random.randint(0, h - size[0])
        left = np.random.randint(0, w - size[1])

        # Apply the occlusion: set the pixel values in the region to the occlusion ID
        occluded_drr[top:top + size[0], left:left + size[1]] = occlusion_ids[i]  # Apply unique occlusion ID

    return occluded_drr, mask
