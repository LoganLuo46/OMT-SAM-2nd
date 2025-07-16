# OMT-SAM
## Description

This repository contains code for training and evaluating an Organ-aware multi-scale medical image segmentation model using text prompt engineering on the Flare 2021 grand challenge dataset. 

## Installation
#### Environment

  - Python version == 3.9.0
    
(1) Clone repository 
```bash
git clone https://github.com/bugman1215/OMT-SAM.git
```
(2) Install requirements
```bash
cd OMT-SAM
pip install -r requirements.txt
```
## Model Training
Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `workdir/SAM/sam_vit_b_01ec64.pth`

Download the [Flare 2021 training dataset](https://zenodo.org/records/5903672), this dataset contains 361 abdomen CT scans, each scan contains an annotation mask with 4 organs: liver ( label=1 ), kidney ( label=2 ), spleen ( label=3 ), and pancreas ( label=4 ).
### Data Preprocessing
Install `cc3d`: `pip install connected-components-3d`

```bash
python pre_CT_2d.py
```
move data/descriptions/organ/descriptions.txt to data/Flare2021npz/imgs/organ/

- adjust CT scans to [soft tissue](https://radiopaedia.org/articles/windowing-ct) window level (40) and width (400)
- max-min normalization
- resample image size to `1024x1024`
- save the pre-processed images and labels as `jpeg/png` files compressed in `npz` files, and organized in the folder structure below:
```
└── data/flare2021/           # Preprocessed dataset directory
│       ├── imgs/               # Image slices
│       │   ├── liver/          # Liver organ images
│               ├── descriptions.txt   # text prompt for segmentation       
│       │   │   ├── train_000.npz
│       │   │   └── ...         # Additional .npz files
│       │   ├── kidney/         # Kidney organ images
│       │   ├── spleen/         # Spleen organ images
│       │   └── pancreas/       # Pancreas organ images
│       └── gts/                # Ground truth masks
│           ├── liver/          # Liver organ masks
│           │   ├── train_000.npz
│           │   └── ...         
│           ├── kidney/
│           ├── spleen/
│           └── pancreas/       

```
### Training on GPUs
```bash
python train_one_gpu.py --ms_features --batch_size [M] --num_workers [N] --device [Device]
```
checkpoints should be saved in work_dir
### Evaluation
Our evaluation metrics include Dice Similarity Coefficient(DSC), Normalized Surface Distance(NSD), and Hausdorff Distance at 95th Percentile(HD95).
```bash
python test_model.py --ms_features --batch_size [M] --num_workers [N] --device [Device] --checkpoint [CHECKPOINT]
```


