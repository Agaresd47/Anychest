import numpy as np

class ImageCombiner:
    """
    A class containing methods for combining soft tissue, rib, spine, and other bone images.
    """

    @staticmethod
    def weighted_addition(soft_tissue: np.ndarray, rib: np.ndarray, spine: np.ndarray, other_bone: np.ndarray,
                          weights: list) -> np.ndarray:
        """
        Combines the images using weighted addition.
        Args:
            soft_tissue: Soft tissue image (NumPy array).
            rib: Rib image (NumPy array).
            spine: Spine image (NumPy array).
            other_bone: Other bone image (NumPy array).
            weights: List of weights for each image [soft_tissue, rib, spine, other_bone].
        Returns:
            Combined image as a NumPy array.
        """
        
        # Ensure all inputs are float32 for weighted operations

        
        combined = (weights[0] * soft_tissue + weights[1] * rib +
                    weights[2] * spine + weights[3] * other_bone)
        return np.clip(combined, 0, 255).astype(np.uint8)

    @staticmethod
    def max_overlay(soft_tissue: np.ndarray, rib: np.ndarray, spine: np.ndarray, other_bone: np.ndarray) -> np.ndarray:
        """
        Combines the images using max overlay (maximum intensity per pixel).
        Args:
            soft_tissue: Soft tissue image (NumPy array).
            rib: Rib image (NumPy array).
            spine: Spine image (NumPy array).
            other_bone: Other bone image (NumPy array).
        Returns:
            Combined image as a NumPy array.
        """
        combined = np.maximum.reduce([soft_tissue, rib, spine, other_bone])
        return combined

    @staticmethod
    def mean_blend(soft_tissue: np.ndarray, rib: np.ndarray, spine: np.ndarray, other_bone: np.ndarray) -> np.ndarray:
        """
        Combines the images by averaging their pixel values.
        Args:
            soft_tissue: Soft tissue image (NumPy array).
            rib: Rib image (NumPy array).
            spine: Spine image (NumPy array).
            other_bone: Other bone image (NumPy array).
        Returns:
            Combined image as a NumPy array.
        """
        combined = (soft_tissue + rib + spine + other_bone) / 4.0
        return np.clip(combined, 0, 255).astype(np.uint8)

