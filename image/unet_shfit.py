import os
from multiprocessing import Pool
from tqdm import tqdm
from image_trans import load_image, load_image_nifti, saving, saving_nifti, apply_transformations
from process import process_transformations, count_transformation_methods, calculate_intervals

transform_combinations = [
    # 1. Geometric + Intensity
    {   "zoom_and_crop": [1.05],
        "rotate_image": [-10],
        "axis_aligned_flip": ["horizontal"],
        
        "adjust_brightness": [5],
        "adjust_contrast": [1.1]
    },

    # 2. Geometric + Noise
    {   "zoom_and_crop": [0.95],
        "rotate_image": [15],
        "axis_aligned_flip": ["vertical"],
        
        "add_gaussian_noise": [0, 0.2],
        "add_speckle_noise": [0.1]
    },

    # 3. Occlusions + Intensity
    {
        "random_occlusions": [2, (30, 30)],
        "adjust_brightness": [-7],
        "apply_gamma_adjustment": [0.9],
        "adjust_contrast": [0.85]
    },

    # 4. Blur + Intensity
    {
        "gaussian_blur": [2],
        "adjust_brightness": [10],
        "apply_gamma_adjustment": [1.2],
        "adjust_contrast": [0.9]
    },

    # 5. Geometric + Advanced Adjustments
    {   "zoom_and_crop": [1.1],
        "rotate_image": [-20],
        "axis_aligned_flip": ["horizontal"],
        
        "bias_field_simulation": [0.2],
        "boundary_smoothing": [3]
    },

    # 6. Occlusions + Noise
    {
        "random_occlusions": [3, (20, 20)],
        "add_gaussian_noise": [0, 0.1],
        "add_speckle_noise": [0.05],
        "adjust_brightness": [-5]
    },

    # 7. Noise + Blur
    {
        "add_gaussian_noise": [0, 0.3],
        "add_speckle_noise": [0.2],
        "gaussian_blur": [3],
        "apply_gamma_adjustment": [1.1]
    },

    # 8. Advanced Adjustments
    {
        "bias_field_simulation": [0.3],
        "boundary_smoothing": [5],
        "histogram_equalization": [True],
        "adjust_brightness": [-8]
    },

    # 9. Geometric + Occlusions
    {   "zoom_and_crop": [1.05],
        "rotate_image": [10],
        "axis_aligned_flip": ["vertical"],
        "random_occlusions": [1, (40, 40)],
        
    },

    # 10. Mixed Transformations
    {   "zoom_and_crop": [0.9],
        "rotate_image": [-15],
        
        "add_gaussian_noise": [0, 0.2],
        "gaussian_blur": [1],
        "adjust_contrast": [1.2]
    }
]

parameter_ranges = {
    # Geometric Transformations
    "rotate_image": (-20, 20),  # Rotation: Range -20° to +20° (evenly sampled)
    "zoom_and_crop": (0.9, 1.1),  # Zoom: Range 0.9x to 1.1x (evenly sampled)
    # Random Occlusions under Geometric trans
    "random_occlusions": {  # Random Occlusions (only size matters here)
        "num_occlusions": (0.00001 , 6),  # number of occlusions:assign whole number
        "size": (20, 40),  # Size: Range 20x20 to 40x40 pixels (evenly sampled)
    },

    # Intensity Transformations
    "adjust_brightness": (-10, 10),  # Brightness: Range -10 to +10 (evenly sampled)
    "adjust_contrast": (0.8, 1.2),  # Contrast: Range 0.8 to 1.2 (evenly sampled)
    "apply_gamma_adjustment": (0.8, 1.2),  # Gamma Correction: Range 0.8 to 1.2 (evenly sampled)

    # Noise Injection
    "add_gaussian_noise": (0.1, 0.3),  # Gaussian Noise: std range 0.1 to 0.3 (evenly sampled)
    "add_speckle_noise": (0.05, 0.2),  # Speckle Noise: Variance range 0.05 to 0.2 (evenly sampled)

    # Blurring/Sharpening
    "gaussian_blur": (1, 3),  # Gaussian Blur: Sigma range 1 to 3 (evenly sampled)
    "gaussian_sharpening": (0.5, 1.5),  # Gaussian Sharpening: Amount range 0.5 to 1.5 (evenly sampled)

    

    # Advanced Adjustments
    "bias_field_simulation": (0.1, 0.3),  # Bias Field Simulation: Max bias 0.1 to 0.3 (evenly sampled)
    "boundary_smoothing": (0.0001, 5),  # Boundary Smoothing: Kernel size 1 to 5 (whole numbers)
}


special_list = [ "axis_aligned_flip", "histogram_equalization"]


def process_image_mask_pair(kwargs):
    """Helper function for parallel processing."""
    img_path = kwargs["img_path"]
    mask_path = kwargs["mask_path"]
    order = kwargs["order"]
    total_samples = kwargs["total_samples"]
    transform_combinations = kwargs["transform_combinations"]
    intervals = kwargs["intervals"]
    special_list = kwargs["special_list"]
    parameter_ranges = kwargs["parameter_ranges"]
    method_counts = kwargs["method_counts"]
    output_img_path = kwargs["output_img_path"]
    output_mask_path = kwargs["output_mask_path"]
    
    # remove 0000 for mask
    mask_path = mask_path.replace("_0000", "")

    # Dynamic loading based on file type
    if img_path.endswith(".nii.gz"):
        img = load_image_nifti(img_path)
        mask = load_image_nifti(mask_path)
        save_func = saving_nifti
    else:
        img = load_image(img_path)
        mask = load_image(mask_path)
        save_func = saving
        
        

    # Generate two updated_combinations for both order and order + total_samples
    updated_combinations_1 = process_transformations(
        order, transform_combinations, intervals, special_list, parameter_ranges, method_counts
    )
    updated_combinations_2 = process_transformations(
        order + total_samples, transform_combinations, intervals, special_list, parameter_ranges, method_counts
    )

    # Combine both updated_combinations
    all_combinations = updated_combinations_1 + updated_combinations_2

    # Counter for naming transformed files
    counter = 0

    # Apply transformations and save results
    for transform_plan in all_combinations:
        counter += 1
        transformed_img, transformed_mask = apply_transformations(img, mask, transform_plan)

        # Generate file names
        base_name = os.path.basename(img_path).split("_0000")[0]
        img_save_name = f"{base_name}_{str(counter).zfill(3)}_0000.nii.gz"
        mask_save_name = f"{base_name}_{str(counter).zfill(3)}.nii.gz"

        # Save transformed image and mask
        save_func(transformed_img, os.path.join(output_img_path, img_save_name))
        save_func(transformed_mask, os.path.join(output_mask_path, mask_save_name))


def augment_dataset(
    master_folder, img_folder, mask_folder, transform_combinations, parameter_ranges, special_list, output_folder
):
    # Paths
    img_path = os.path.join(master_folder, img_folder)
    mask_path = os.path.join(master_folder, mask_folder)

    # Verify consistency between image and mask folders
    img_files = sorted(os.listdir(img_path))
    mask_files = sorted(os.listdir(mask_path))

    # Remove '_0000' suffix from image filenames before comparison
    img_files_modified = [file.replace("_0000", "") for file in img_files]

    # Compare the modified image filenames with the mask filenames
    if set(img_files_modified) != set(mask_files):
        mismatched = set(img_files_modified).symmetric_difference(set(mask_files))
        raise ValueError(f"Image and mask folders do not match. Mismatched files: {mismatched}")


    # Count transformation methods
    method_counts = count_transformation_methods(transform_combinations)

    # Calculate intervals
    total_samples = len(img_files) // 2
    intervals = calculate_intervals(method_counts, total_samples, parameter_ranges, special_list)
    
    # Hard code the methods
    special_list.append("random_occlusions")
    special_list.append("add_gaussian_noise")

    # Process only half of the dataset
    half_dataset = img_files[: total_samples]

    # Create output folders for augmented data
    output_img_path = os.path.join(output_folder, img_folder)
    output_mask_path = os.path.join(output_folder, mask_folder)
    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)

    # Prepare inputs for multiprocessing
    kwargs_list = []
    for idx, img_file in enumerate(half_dataset, start=1):
        img_file_path = os.path.join(img_path, img_file)
        mask_file_path = os.path.join(mask_path, img_file)

        # Add arguments for multiprocessing
        kwargs_list.append({
            "img_path": img_file_path,
            "mask_path": mask_file_path,
            "order": idx,
            "total_samples": total_samples,
            "transform_combinations": transform_combinations,
            "intervals": intervals,
            "special_list": special_list,
            "parameter_ranges": parameter_ranges,
            "method_counts": method_counts,
            "output_img_path": output_img_path,
            "output_mask_path": output_mask_path,
        })

    # Process with multiprocessing
    with Pool() as pool:
        list(tqdm(pool.imap(process_image_mask_pair, kwargs_list), total=len(kwargs_list)))

    print(f"Augmentation complete! Results saved to {output_img_path} and {output_mask_path}")


# Example Usage
augment_dataset(
    master_folder="/data/bruce/unet/Dataset002_AnychestTry",
    img_folder="imagesTr",
    mask_folder="labelsTr",
    transform_combinations=transform_combinations,
    parameter_ranges=parameter_ranges,
    special_list=special_list,
    output_folder="/data/bruce/unet/Dataset002_AnychestTryAug"
)
