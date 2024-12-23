import numpy as np

def add_gaussian_noise(drr: np.ndarray, mask: np.ndarray, mean: float, std_dev: float) -> tuple:
    """
    Adds Gaussian noise to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        mean: Mean of the Gaussian noise.
        std_dev: Standard deviation of the Gaussian noise.
    Returns:
        Tuple of transformed DRR and unchanged mask (NumPy arrays).
    """
    noise = np.random.normal(mean, std_dev, drr.shape).astype(np.float32)
    noisy_drr = np.clip(drr + noise, 0, 255).astype(np.uint8)
    return noisy_drr, mask

def add_rician_noise(drr: np.ndarray, mask: np.ndarray, sigma: float) -> tuple:
    """
    Adds Rician noise to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        sigma: Standard deviation of the Rician noise.
    Returns:
        Tuple of transformed DRR and unchanged mask (NumPy arrays).
    """
    noise_real = np.random.normal(0, sigma, drr.shape).astype(np.float32)
    noise_imag = np.random.normal(0, sigma, drr.shape).astype(np.float32)
    noisy_drr = np.sqrt((drr + noise_real) ** 2 + noise_imag ** 2)
    noisy_drr = np.clip(noisy_drr, 0, 255).astype(np.uint8)
    return noisy_drr, mask

def add_speckle_noise(drr: np.ndarray, mask: np.ndarray, variance: float) -> tuple:
    """
    Adds Speckle noise to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
        variance: Variance of the speckle noise.
    Returns:
        Tuple of transformed DRR and unchanged mask (NumPy arrays).
    """
    noise = np.random.normal(0, np.sqrt(variance), drr.shape).astype(np.float32)
    noisy_drr = np.clip(drr + drr * noise, 0, 255).astype(np.uint8)
    return noisy_drr, mask

def add_poisson_noise(drr: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Adds Poisson noise to the DRR image.
    Args:
        drr: The input DRR image (NumPy array).
        mask: The corresponding mask (NumPy array).
    Returns:
-    """
    noisy_drr = np.random.poisson(drr).astype(np.float32)
    noisy_drr = np.clip(noisy_drr, 0, 255).astype(np.uint8)
    return noisy_drr, mask
