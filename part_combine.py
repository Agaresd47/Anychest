import os
import nibabel as nib
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm


def load_nifti(file_path):
    """
    Load a NIfTI file and return the image data, affine, and header.
    """
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header

def save_nifti(data, affine, header, output_path):
    """
    Save data as a NIfTI file, retaining the original affine and header.
    """
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, output_path)
    
def load_image(image_path):
    """
    Load a JPG image and convert it to a numpy array.
    """
    try:
        img = Image.open(image_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: File not found - {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def combine_images(images, method='max'):
    """
    Combine multiple images using a specified method.
    Default method is 'max' to get the highest intensity at each pixel.
    """
    combined_image = None

    for img in images:
        if img is None:
            continue  # Skip if an image failed to load
        if combined_image is None:
            combined_image = img
        else:
            if method == 'max':
                combined_image = np.maximum(combined_image, img)
            elif method == 'average':
                combined_image = (combined_image + img) / 2

    return combined_image

def save_image(image_data, output_path):
    """
    Save a numpy array as a JPG image.
    """
    try:
        img = Image.fromarray(image_data.astype(np.uint8))
        img.save(output_path)
        #print(f"Saved combined image: {output_path}")
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        
def combine_masks(mask_files, exclude=None):
    """
    Combine multiple NIfTI masks into a single mask.
    If exclude is specified, it removes the specified mask from the combination.
    """
    combined_mask = None
    affine = None
    header = None

    for mask_file in mask_files:
        if exclude and os.path.basename(mask_file) == exclude:
            continue

        # Load the NIfTI file
        img = nib.load(mask_file)
        mask_data = img.get_fdata()

        if combined_mask is None:
            combined_mask = mask_data
            affine = img.affine
            header = img.header
        else:
            combined_mask = np.maximum(combined_mask, mask_data)  # Combine by taking the max

    return combined_mask, affine, header

def save_combined_mask(mask_data, affine, header, output_path):
    """
    Save the combined mask data to a new NIfTI file.
    """
    new_img = nib.Nifti1Image(mask_data, affine, header)
    nib.save(new_img, output_path)
        
def combine_and_save_masks(patient_id, source_folder, output_folder):
    """
    Combine spine, rib, and bones masks and save the results.
    """
    # Define file paths for each mask category (exclude 'costal_cartilages.nii.gz' for bones)
    categories = {
        'spine': os.path.join(source_folder, 'Spine', f'{patient_id}_total'),
        'rib': os.path.join(source_folder, 'Rib_clean', f'{patient_id}_total'),
        'bones': os.path.join(source_folder, 'Bones', f'{patient_id}_total')
    }

    # Create the subfolder for this patient if it doesn't exist
    patient_folder = os.path.join(output_folder, patient_id)
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    # Process each category
    for category, folder in categories.items():
        #print(f"Processing {category}...")
        
        # List all files in the current category folder
        mask_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii.gz')]

        if category == 'bones':
            # Exclude costal_cartilages for the bones combination
            combined_mask, affine, header = combine_masks(mask_files, exclude='costal_cartilages.nii.gz')
        else:
            combined_mask, affine, header = combine_masks(mask_files)

        # Save the combined mask
        output_file = os.path.join(patient_folder, f'{patient_id}_{category}_mask.nii.gz')
        save_combined_mask(combined_mask, affine, header, output_file)
        #print(f"Saved combined {category} mask: {output_file}")

def process_masks_and_ct(patient_id, source_folder, output_folder, padding_value=5):
    """
    Process combined masks to extract real bone, rib, spine intensities
    and create a soft tissue mask.
    """
    # File paths for the combined masks
    combined_paths = {
        "bones": os.path.join(output_folder, patient_id, f"{patient_id}_bones_mask.nii.gz"),
        "rib": os.path.join(output_folder, patient_id, f"{patient_id}_rib_mask.nii.gz"),
        "spine": os.path.join(output_folder, patient_id, f"{patient_id}_spine_mask.nii.gz")
    }
    ct_file_path = os.path.join(source_folder, "CT_no_bed", f"{patient_id}.nii.gz")

    # Output paths for the real masks and soft tissue
    output_paths = {
        "bones": os.path.join(output_folder, patient_id, f"{patient_id}_bones_real.nii.gz"),
        "rib": os.path.join(output_folder, patient_id, f"{patient_id}_rib_real.nii.gz"),
        "spine": os.path.join(output_folder, patient_id, f"{patient_id}_spine_real.nii.gz"),
        "soft_tissue": os.path.join(output_folder, patient_id, f"{patient_id}_soft_tissue.nii.gz")
    }
    

    # Create the patient subfolder if it doesn't exist
    patient_folder = os.path.join(output_folder, patient_id)
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    # Load CT scan
    #print(f"Loading CT: {ct_file_path}")
    ct_data, ct_affine, ct_header = load_nifti(ct_file_path)

    # Initialize soft tissue data as the original CT data
    soft_tissue_data = ct_data.copy()

    # Process each combined mask (bones, rib, spine)
    for mask_type, mask_path in combined_paths.items():
        #print(f"Loading Combined Mask: {mask_path}")
        mask_data, mask_affine, mask_header = load_nifti(mask_path)

        # Extract real intensities by applying the mask to the CT data
        real_mask_data = np.where(mask_data > 0, ct_data, -999.9)

        # Save the real mask with CT intensities
        save_nifti(real_mask_data, ct_affine, ct_header, output_paths[mask_type])
        #print(f"Saved Real {mask_type.capitalize()} Mask: {output_paths[mask_type]}")

        # Hollow out the CT in the mask areas for the soft tissue mask
        soft_tissue_data = np.where(mask_data > 0, padding_value, soft_tissue_data)

    # Save the soft tissue mask
    save_nifti(soft_tissue_data, ct_affine, ct_header, output_paths["soft_tissue"])
    #print(f"Saved Soft Tissue Mask: {output_paths['soft_tissue']}")
    



def process_and_combine_images(input_folder, output_folder, patient_id, method='max'):
    """
    Process the images (spine, rib, bone, soft tissue), combine them, and save the result.
    """
     # Paths to the images
    image_paths = {
        "spine": os.path.join(input_folder, f"{patient_id}_spine_real.jpg"),
        "bones": os.path.join(input_folder, f"{patient_id}_bones_real.jpg"),
        "rib": os.path.join(input_folder, f"{patient_id}_rib_real.jpg"),
        "soft_tissue": os.path.join(input_folder, f"{patient_id}_soft_tissue.jpg")
    }

    # Load images
    images = [load_image(path) for path in image_paths.values()]

    # Combine images
    combined_image = combine_images(images, method)

    
    if combined_image is not None:
        # Output path
        output_path = os.path.join(output_folder, f"{patient_id}_combined_drr.jpg")
        save_image(combined_image, output_path)
        return combined_image
    else:
        print("No images were successfully combined. Output skipped.")
        
def process_patient(patient_id, source_folder, output_folder, padding_value=5):
    """
    Wrapper function to combine masks and process CT/masks for each patient.
    """
    # Step 1: Combine and save masks
    combine_and_save_masks(patient_id, source_folder, output_folder)
    
    # Step 2: Process masks and CT
    process_masks_and_ct(patient_id, source_folder, output_folder, padding_value)

def collect_patient_ids(source_folder):
    """
    Collect all patient IDs from the source folder by reading file names.
    """
    patient = os.path.join(source_folder, "CT_no_bed")
    patient_ids = []
    for file_name in os.listdir(patient):
        if file_name.endswith(".nii.gz"):
            patient_id = file_name.replace(".nii.gz", "")
            patient_ids.append(patient_id)
    return patient_ids

# Wrapper function
def process_patient_wrapper(args):
    process_patient(*args)
    return 0

def PP_process(source_folder, output_main_folder, padding_value=5, num_workers=10):
    """
    Main function to process all patients in parallel.
    """
    # Step 1: Collect all patient IDs from the source folder
    patient_ids = collect_patient_ids(source_folder)
    print(f"Found {len(patient_ids)} patients for processing.")

    # Step 2: Prepare arguments for parallel processing
    args_list = [
        (patient_id, source_folder, os.path.join(output_main_folder, patient_id), padding_value)
        for patient_id in patient_ids
    ]
    
    print(f"Processing {len(args_list)} patients with {num_workers} workers...")
    
    # Step 3: Use multiprocessing to process the patients in parallel
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_patient_wrapper, args_list), total=len(args_list)))

if __name__ == "__main__":
     # Define paths
    source_folder = "/home/zifei/data/CT_study/drr_100/First_100"
    output_main_folder = "/home/zifei/data/CT_study/drr_100/First_100_Sep"
    
    # Call the main function to start processing
    PP_process(source_folder, output_main_folder, padding_value=5, num_workers=10)
