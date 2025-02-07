from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np

def find_surface(array):
    # Create a binary mask of the structure
    binary_mask = array >= -100  # Assume non-zero values are part of the structure

    # Dilate and erode to find the surface
    outer_surface = binary_dilation(binary_mask) & ~binary_mask
    inner_surface = binary_mask & ~binary_erosion(binary_mask)

    # Combine to get the surface mask
    surface_mask = outer_surface | inner_surface

    return surface_mask, outer_surface, inner_surface

def find_inner_surface(binary_mask, inner_percentage=3):
    # Estimate the number of layers for the given percentage
    total_thickness = max(binary_mask.shape)
    inner_layers = int(total_thickness * inner_percentage / 100)

    # Erode to find the inner region
    inner_surface = binary_mask & ~binary_erosion(binary_mask, iterations=inner_layers)

    return inner_surface


def adjust_surface_intensity(array, surface_mask, inner_surface, adjustment_factor=1.1, transition="linear"):
    # Initialize adjusted array
    adjusted_array = array.copy()

    # Apply enhancement to the outer surface
    adjusted_array[surface_mask] *= adjustment_factor

    # Apply a transition for the inner surface
    if transition == "linear":
        for i, layer in enumerate(inner_surface):
            factor = 1 + (adjustment_factor - 1) * (1 - i / len(inner_surface))
            adjusted_array[layer] *= factor

    return adjusted_array
def enhance_surface(array, adjustment_factor=1.1, inner_percentage=3):
    # Step 1: Find surface voxels
    surface_mask, outer_surface, inner_surface = find_surface(array)

    # Step 2: Find the inner surface region
    binary_mask = array > 0
    inner_surface = find_inner_surface(binary_mask, inner_percentage=inner_percentage)

    # Step 3: Apply intensity adjustment with transition
    enhanced_array = adjust_surface_intensity(array, surface_mask, inner_surface, adjustment_factor)

    return enhanced_array

# Example usage
array = np.random.rand(100, 100, 100)  # Replace with your array
enhanced_array = enhance_surface(array, adjustment_factor=1.1, inner_percentage=3)
