import numpy as np
import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import nibabel as nib

from noise_trans import add_gaussian_noise, add_rician_noise, add_speckle_noise, add_poisson_noise
from geometric_trans import (
    spatial_cropping, rotate_image, axis_aligned_flip, perspective_transform,
    zoom_and_crop, random_occlusions
)
from advanced_image_adjustments import (
    bias_field_simulation, histogram_equalization, boundary_smoothing
)
from blur_sharpen_trans import gaussian_blur, gaussian_sharpening
from intensity_trans import apply_gamma_adjustment, adjust_brightness, adjust_contrast, adjust_contrast_with_curve
from image_combine import ImageCombiner



# Master API
def apply_transformations(drr: np.ndarray, mask: np.ndarray, transform_plan: dict) -> tuple:
    """
    Applies a series of transformations to the DRR image and mask in a specified order.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        transform_plan: A dictionary where the key is the method name and the value is a list of arguments.
    Returns:
        Tuple of transformed DRR and mask (NumPy arrays).
    """
    # Combine global and imported methods
    all_methods = {**globals(), **sys.modules['noise_trans'].__dict__}

    for method_name, args in transform_plan.items():
        # Resolve the method dynamically
        method = all_methods.get(method_name)
        if method is None:
            raise ValueError(f"Transformation method '{method_name}' not found.")
        drr, mask = method(drr, mask, *args)
    return drr, mask

# Image Loading Function
def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image as a grayscale (L mode) NumPy array.
    Args:
        image_path: Path to the image file.
    Returns:
        NumPy array of the grayscale image.
    """
    image = Image.open(image_path).convert("L")
    return np.array(image)

# Empty Shell for Post-Processing (PP)
def post_processing(drr: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Empty shell for post-processing the transformed image and mask.
    Args:
        drr: The transformed DRR image (NumPy array).
        mask: The transformed mask (NumPy array).
    Returns:
        Tuple of processed DRR and mask (NumPy arrays).
    """
    # Placeholder for future implementation
    return drr, mask

def combine_images(soft_tissue: np.ndarray, rib: np.ndarray, spine: np.ndarray, other_bone: np.ndarray,
                   combine_plan: dict) -> np.ndarray:
    """
    Combines the four images into a single image using the specified plan.
    Args:
        soft_tissue: Soft tissue image (NumPy array).
        rib: Rib image (NumPy array).
        spine: Spine image (NumPy array).
        other_bone: Other bone image (NumPy array).
        combine_plan: Dictionary where key is the method name, and value is a list of arguments for that method.
    Returns:
        Combined image as a NumPy array.
    """
    combined = None
    for method_name, args in combine_plan.items():
        # Dynamically call the combination method
        method = getattr(ImageCombiner, method_name, None)
        if method is None:
            raise ValueError(f"Combination method '{method_name}' not found.")
        combined = method(soft_tissue, rib, spine, other_bone, *args)
    return combined


def saving(image: np.ndarray, save_path: str) -> None:
    """
    Saves the given image to the specified path.
    Args:
        image: The image to be saved (NumPy array).
        save_path: Path where the image should be saved.
    Returns:
        None
    """
    # Convert NumPy array to PIL Image
    image_to_save = Image.fromarray(image)
    
    # Save the image
    image_to_save.save(save_path)
    
# Load NIfTI image
def load_image_nifti(image_path: str) -> np.ndarray:
    # Load the NIfTI file using nibabel
    nifti_image = nib.load(image_path)
    # Return the image data as a numpy array
    return nifti_image.get_fdata()

# Save NIfTI image
def saving_nifti(image: np.ndarray, save_path: str):
    # Create a new NIfTI image from the numpy array
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))  # Default affine matrix (identity)
    # Save the NIfTI image to the specified path
    nib.save(nifti_image, save_path)



# Example Usage
if __name__ == "__main__":
    # Dummy example input
    drr = np.ones((256, 256), dtype=np.uint8) * 128  # Example DRR
    mask = np.zeros((256, 256), dtype=np.uint8)  # Example mask

    # Define the transformation plan
    transform_plan = {
        "add_gaussian_noise": [0.0, 15.0],  # Add Gaussian noise with mean=0.0 and std_dev=15.0
        "rotate_image": [15],  # Rotate by 15 degrees
        "apply_gamma_adjustment": [1.5]  # Adjust gamma with gamma=1.5
    }

    # Apply transformations
    transformed_drr, transformed_mask = apply_transformations(drr, mask, transform_plan)

    # Visualize results (optional)
    plt.subplot(1, 2, 1)
    plt.title("Transformed DRR")
    plt.imshow(transformed_drr, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Transformed Mask")
    plt.imshow(transformed_mask, cmap="gray")
    plt.show()
