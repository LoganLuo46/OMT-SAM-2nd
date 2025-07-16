# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
join = os.path.join
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.SurfaceDice import compute_dice_coefficient
from open_clip import create_model_from_pretrained
from get_clip_embedding1 import ModifiedCLIPModel
from get_clip_embedding1 import create_modified_clip_model
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from test_data import OrganSliceDataset, ORGANS



import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder.
Freeze prompt image encoder.
"""
# Helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
def train_val_split(dataset, val_ratio=0.2):
    n = len(dataset.slice_map)          # 切片级索引
    idx = np.random.permutation(n)
    n_val = int(n * val_ratio)
    val_idx  = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    return train_ds, val_ds


def build_loaders(root_dir, batch_size=4, num_workers=4):
    train_sets, val_sets, loaders = {}, {}, {}
    for oid in ORGANS.keys():
        try:
            full_ds = OrganSliceDataset(root_dir, organ_id=oid)
        except RuntimeError:
            continue
        train_ds, val_ds = train_val_split(full_ds, val_ratio=0.2)
        train_sets[oid], val_sets[oid] = train_ds, val_ds
        loaders[oid] = {
            "train": DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True,  num_workers=num_workers,
                                pin_memory=True),
            "val":   DataLoader(val_ds,   batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        }
    return loaders

# MedSAM model
class MedSAM(nn.Module):
    def __init__(self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        use_clip=False,

    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_clip = use_clip

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        for name, param in self.image_encoder.named_parameters():
            if "neck_list" in name or "neck" in name:
                param.requires_grad = True

        if self.use_clip:
            clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self.embed_dim = 256
            num_heads = 8
            self.clip_model, self.clip_preprocess = create_model_from_pretrained(clip_model_name)
            self.modified_clip_model = ModifiedCLIPModel(self.clip_model, self.embed_dim, num_heads)
        else:
            self.modified_clip_model = None
            self.clip_model = None
            self.clip_preprocess = None

    def get_clip_embeddings(self, images, text_inputs):

        if not self.use_clip:
            return None

        with torch.no_grad():
            processed_images = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    # 若单通道，先补成 3 通道；再乘 255 → uint8
                    if image.shape[0] == 1:
                        image = image.repeat(3, 1, 1)
                    image = (image * 255).clamp(0, 255).byte().cpu()
                    image = T.ToPILImage()(image)
                processed_images.append(self.clip_preprocess(image).unsqueeze(0))

            clip_inputs = torch.cat(processed_images, dim=0)

        attn_output = self.modified_clip_model(clip_inputs.to(images.device), text_inputs.to(images.device))
        clip_image_embeddings = attn_output
        clip_prompt_embeddings = clip_image_embeddings.view(
            clip_image_embeddings.size(0),
            1,
            self.embed_dim,
        )
        return clip_prompt_embeddings

    def forward(self, image, box, text_input):

        image_embedding = self.image_encoder(image)
        device = image_embedding[0].device if isinstance(image_embedding, list) else image_embedding.device

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )

        # !!!!!!
        clip_embeddings = None
        if self.use_clip:
            clip_embeddings = self.get_clip_embeddings(image, text_input).to(device)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            clip_prompt_embeddings=clip_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

from collections import defaultdict

from collections import defaultdict
import torch

loss_fn = lambda p,t: (
    monai.losses.DiceLoss(sigmoid=True, squared_pred=True)(p, t) +
    nn.BCEWithLogitsLoss()(p, t.float())
)


def train_epoch(model, loaders, optimizer, loss_fn, device, use_amp=False):
    model.train();
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epoch_loss, epoch_dice, n_batch = 0, 0, 0

    for oid, dl in loaders.items():
        for imgs, gts, bboxes, names, text_ids in dl["train"]:
            imgs, gts = imgs.to(device), gts.to(device)
            text_ids   = text_ids.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(imgs, bboxes.numpy(), text_ids)
                loss  = loss_fn(preds, gts)           # Dice+BCE 已封装
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            dice = compute_dice_coefficient(gts.cpu().numpy(),
                                            (preds>0.5).cpu().numpy())
            epoch_loss += loss.item(); epoch_dice += dice; n_batch += 1

    return epoch_loss/n_batch, epoch_dice/n_batch



# Main training script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="data/npy/CT_Abd")
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--load_pretrain", type=bool, default=True, help="Load pretrain model")
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="Learning rate")
    parser.add_argument("-use_wandb", type=bool, default=False, help="Use wandb for training log")
    parser.add_argument("-use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    ### new params
    parser.add_argument("--ms_features", action="store_true")
    parser.add_argument("--one_neck", action="store_true")
    parser.add_argument("--use_clip", type=bool, default=True,help="Whether to use CLIP model for text and image prompt fusion")

    args = parser.parse_args()
    print("Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared.")
    join = os.path.join
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    use_clip_str = "_use_clip" if args.use_clip else "_no_clip"
    model_save_path = join(args.work_dir,
                           args.task_name + f"_MS{args.ms_features}" + f"_oneneck{args.one_neck}" + use_clip_str + "_" + run_id)

    #args = parser.parse_args()
    device = torch.device(args.device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created directory: {model_save_path}")
    
    ### ======== modify !!!!!!!!!!!!!!!!!!!! ======== ###
    sam_model = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        ms_features=args.ms_features,
        one_neck=args.one_neck,
    )
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        use_clip=args.use_clip,
    )

    loaders = build_loaders(args.tr_npy_path,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        medsam_model = medsam_model.to(args.device)
    medsam_model.train()

    optimizer = torch.optim.AdamW(
        medsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    ce_loss = nn.BCEWithLogitsLoss()

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Ensure 'state_dict' is extracted if the checkpoint contains additional metadata
    if 'model' in checkpoint:
        state_dict = checkpoint['model']  # Adjust this key based on the actual checkpoint structure
    else:
        state_dict = checkpoint

# Load the extracted state_dict into the model
    medsam_model.load_state_dict(state_dict, strict=False)

# Optionally, extract the starting epoch if available
    start_epoch = checkpoint.get('epoch', 0)
    start_epoch+=1

    print(f"Checkpoint loaded successfully. Starting from epoch {start_epoch}.")
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        #Store training and validation metrics
    train_losses = []
    train_accuracy = []

    val_losses = {ORGANS[oid]: [] for oid in loaders.keys()}
    val_accuracies = {ORGANS[oid]: [] for oid in loaders.keys()}
    # num_epochs=100
    best_loss = 1e10
    iter_num=0
        
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0
        epoch_dice = 0
        for oid, dl in loaders.items():
            for image, gt2D, boxes, _, text_input in dl["train"]:
                optimizer.zero_grad()
                image, gt2D = image.to(device), gt2D.to(device)
                boxes_np = boxes.numpy()
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = medsam_model(image, boxes_np, text_input)
                        loss = loss_fn(pred, gt2D)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    pred = medsam_model(image, boxes_np, text_input)
                    loss = loss_fn(pred, gt2D)
                    loss.backward();
                    optimizer.step()

                dice = compute_dice_coefficient(gt2D.cpu().numpy(),
                                                (pred > 0.5).cpu().numpy())
                epoch_loss += loss.item()
                epoch_dice += dice

                iter_num += 1

        total_batches = sum(len(dl["train"]) for dl in loaders.values())
        epoch_loss /= total_batches
        epoch_dice /= total_batches
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_dice)

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss}, {"epoch_dice": epoch_dice})# Validation loop

        medsam_model.eval()
        val_results = defaultdict(dict)
        with torch.no_grad():

            for oid, dl in loaders.items():
                organ = ORGANS[oid]
                val_loader = dl["val"]

                total_loss, total_dice, count = 0, 0, 0

                # use tqdm to show progress bar
                for image, gt, boxes, _, text_input in tqdm(val_loader, desc=f"Validating {organ}", leave=False):
                    bboxes = boxes.numpy()
                    image, gt = image.to(device), gt.to(device)

                    # model prediction
                    medsam_pred = medsam_model(image, bboxes, text_input)
                    loss = seg_loss(medsam_pred, gt) + ce_loss(medsam_pred, gt.float())
                    dice = compute_dice_coefficient(
                        gt.cpu().numpy(), (medsam_pred > 0.5).cpu().numpy()
                    )

                    # total loss and dice
                    total_loss += loss.item()
                    total_dice += dice
                    count += 1

                # compute average loss and dice on validation set
                val_results[organ]["loss"] = total_loss / count
                val_results[organ]["dice"] = total_dice / count
                val_losses[organ].append(val_results[organ]["loss"])
                val_accuracies[organ].append(val_results[organ]["dice"])
                print(f"{organ.capitalize()} Validation Complete: Loss: {val_results[organ]['loss']:.4f}, Dice: {val_results[organ]['dice']:.4f}")
           
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, train_Loss: {epoch_loss}, train_accuracy : {epoch_dice}'
        )
        # Logging metrics
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        total_val_loss = 0 
        for loss in val_losses.values():
            total_val_loss += loss[-1] / len(val_losses)
        
        ## save the best model
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print("saved a better ckpt.")

        print("val loss:", total_val_loss)
        for organ, metrics in val_results.items():
            print(f"  {organ.capitalize()} - Val Loss: {metrics['loss']:.4f}, Dice: {metrics['dice']:.4f}")
        
        # %% plot loss
    
        # Update loss plots
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        for organ, losses in val_losses.items():
            plt.plot(losses, label=f"Val Loss ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy, label="Train accuracy")
        for organ, accuracies in val_accuracies.items():
            plt.plot(accuracies, label=f"Val Dice ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Validation Dice Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(join(model_save_path, "training_curves.png"))
        plt.close()

if __name__ == "__main__":
    main()
