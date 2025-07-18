# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# convert nii image to npz files, including original image and corresponding masks
modality = "CT"
anatomy = "Abd"  # anantomy + dataset name
img_name_suffix = "_0000.nii.gz"
gt_name_suffix = ".nii.gz"
prefix = modality + "_" + anatomy + "_"

nii_path = "/Users/crisp/Downloads/FLARE22_LabeledCase50/images"   # 影像 *.nii.gz
gt_path  = "/Users/crisp/Downloads/FLARE22_LabeledCase50/labels"   # 标签 *.nii.gz
npy_path = "data/npy/" + prefix[:-1]
os.makedirs(npy_path, exist_ok=True)

image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 400  # only for CT images

# %% save preprocessed images and masks as npz files
for name in tqdm(names[:40]):  # use the remaining 10 cases for validation
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # remove label ids
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices

    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]

        # save the image and ground truth as nii files for sanity check;
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        # save the each CT image as npz file
        np.savez_compressed(
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + ".npz"),
            imgs=img_roi.astype(np.uint8),  # (Nz, H, W)
            gts=gt_roi.astype(np.uint8),  # (Nz, H, W)
            spacing=np.array(img_sitk.GetSpacing(), dtype=np.float32)
        )
