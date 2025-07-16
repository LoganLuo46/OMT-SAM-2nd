# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
import os
from skimage import transform
from tqdm import tqdm
import cc3d
import logging

join = os.path.join

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
modality = "CT"
organs = ["liver", "kidney", "spleen", "pancreas"]
label_ids = {1: "liver", 2: "kidney", 3: "spleen", 4: "pancreas"}
image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400

# Paths
nii_path = "data/Flare2021/Images"
gt_path = "data/Flare2021/TrainingMask"
base_output_path = "data/Flare2021npz"  
img_name_suffix = "_0000.nii.gz"
gt_name_suffix = ".nii.gz"

def setup_directories(base_path, organs):
    for organ in organs:
        os.makedirs(join(base_path, "imgs", organ), exist_ok=True)
        os.makedirs(join(base_path, "gts", organ), exist_ok=True)

def process_files(nii_path, gt_path, output_path, num_files=361):
    setup_directories(output_path, organs)
    
    if not os.path.exists(gt_path):
        logger.error(f"Ground truth path does not exist: {gt_path}")
        return
    if not os.path.exists(nii_path):
        logger.error(f"Image path does not exist: {nii_path}")
        return
    
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(gt_name_suffix)])
    logger.info(f"Found {len(gt_files)} ground truth files in {gt_path}: {gt_files[:5]}...")
    
    names = []
    for gt_name in gt_files:
        base_name = gt_name.split(gt_name_suffix)[0]
        img_name = f"{base_name}{img_name_suffix}"
        img_full_path = join(nii_path, img_name)
        if os.path.exists(img_full_path):
            names.append(gt_name)
        else:
            logger.warning(f"Image not found for {gt_name}: {img_full_path}")
    
    logger.info(f"After sanity check, valid pairs: {len(names)}. First few: {names[:5]}")
    
    if not names:
        logger.error("No valid file pairs found. Check paths and file naming.")
        return
    
    for name in tqdm(names[:num_files]):
        try:
            base_filename = name.split(gt_name_suffix)[0]
            image_name = f"{base_filename}{img_name_suffix}"
            gt_full_path = join(gt_path, name)
            img_full_path = join(nii_path, image_name)
            
            gt_sitk = sitk.ReadImage(gt_full_path)
            gt_data = np.uint8(sitk.GetArrayFromImage(gt_sitk))
            gt_data = cc3d.dust(gt_data, threshold=voxel_num_thre3d, 
                              connectivity=26, in_place=True)
            
            img_sitk = sitk.ReadImage(img_full_path)
            img_data = sitk.GetArrayFromImage(img_sitk)
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            img_data = np.clip(img_data, lower_bound, upper_bound)
            img_data = ((img_data - np.min(img_data)) / 
                       (np.max(img_data) - np.min(img_data)) * 255.0)
            img_data = np.uint8(img_data)
            
            z_index = np.unique(np.where(gt_data > 0)[0])
            if len(z_index) == 0:
                logger.warning(f"No valid slices found in {name}")
                continue
                
            img_slices_by_organ = {organ: [] for organ in organs}
            gt_slices_by_organ = {organ: [] for organ in organs}
            
            for slice_idx in z_index:
                img_slice = img_data[slice_idx]
                img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)
                img_resized = transform.resize(
                    img_3c, (image_size, image_size), order=3,
                    preserve_range=True, mode="constant", anti_aliasing=True
                )
                img_resized = (img_resized - img_resized.min()) / np.clip(
                    img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
                )
                
                gt_slice = gt_data[slice_idx]
                gt_resized = transform.resize(
                    gt_slice, (image_size, image_size), order=0,
                    preserve_range=True, mode="constant", anti_aliasing=False
                )
                gt_resized = np.uint8(gt_resized)
                gt_resized = cc3d.dust(gt_resized, threshold=voxel_num_thre2d,
                                     connectivity=8, in_place=True)
                
                for label_id, organ in label_ids.items():
                    organ_mask = (gt_resized == label_id).astype(np.uint8)
                    if np.any(organ_mask):
                        img_slices_by_organ[organ].append(img_resized)
                        gt_slices_by_organ[organ].append(organ_mask)
            
            for organ in organs:
                if img_slices_by_organ[organ]:
                    img_stack = np.stack(img_slices_by_organ[organ], axis=0)
                    np.savez_compressed(
                        join(output_path, "imgs", organ, f"{base_filename}.npz"),
                        imgs=img_stack  # Key "imgs"
                    )
                    logger.debug(f"Saved {organ} images with shape {img_stack.shape}")
                
                if gt_slices_by_organ[organ]:
                    gt_stack = np.stack(gt_slices_by_organ[organ], axis=0)
                    np.savez_compressed(
                        join(output_path, "gts", organ, f"{base_filename}.npz"),
                        gts=gt_stack  # Key "gts"
                    )
                    logger.debug(f"Saved {organ} labels with shape {gt_stack.shape}")
                    
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue

if __name__ == "__main__":
    process_files(nii_path, gt_path, base_output_path, num_files=361)
    logger.info("Processing complete")