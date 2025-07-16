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
from torch.utils.data import Dataset, DataLoader, Subset
import json
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
from test_data_2nd_version import OrganSliceDataset, ORGANS
import wandb



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


def split_organs_by_count(dataset, train_count=8):
    n = len(dataset.slice_map)
    idx = np.random.permutation(n)
    train_idx = idx[:train_count]
    val_idx = idx[train_count:]
    return train_idx, val_idx



def build_loaders(root_dir, tokenizer,train_organs, test_organs, batch_size=4, num_workers=4, prompt_file="organ_prompts_ten.txt", val_ratio=0.2):
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for oid in train_organs:
        full_ds = OrganSliceDataset(root_dir, organ_id=oid,
                                    tokenizer=tokenizer, prompt_file=prompt_file)
        n = len(full_ds.slice_map)
        idx = np.random.permutation(n)
        split = int(n * (1 - val_ratio))
        tr_ds = Subset(full_ds, idx[:split])
        val_ds = Subset(full_ds, idx[split:])

        train_loaders[oid] = DataLoader(tr_ds, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loaders[oid] = DataLoader(val_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers, pin_memory=True)

    for oid in test_organs:
        test_ds = OrganSliceDataset(root_dir, organ_id=oid,
                                    tokenizer=tokenizer, prompt_file=prompt_file)
        test_loaders[oid] = DataLoader(test_ds, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)

    return train_loaders, val_loaders, test_loaders


# MedSAM model
class MedSAM(nn.Module):
    def __init__(self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        use_clip=False,
        clip_variant="biomedclip",
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
            if clip_variant == "biomedclip":
                clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            elif clip_variant == "clip":
                clip_model_name = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
            elif clip_variant == "bioclip":
                clip_model_name = "hf-hub:imageomics/bioclip"
            print(f"[MedSAM] clip_variant: {clip_variant}, clip_model_name: {clip_model_name}")
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
                loss  = loss_fn(preds, gts.float())           # Dice+BCE 已封装
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
    parser.add_argument("--checkpoint", type=str, default="./work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--load_pretrain", type=bool, default=True, help="Load pretrain model")
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir1")
    parser.add_argument("-num_epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-num_workers", type=int, default=8)
    parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="Learning rate")
    parser.add_argument("-use_wandb", action="store_true", default=False, help="Use wandb for training log")
    parser.add_argument("-use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    ### new params
    parser.add_argument("--ms_features", action="store_true")
    parser.add_argument("--one_neck", action="store_true")
    parser.add_argument("--use_clip", type=bool, default=True,help="Whether to use CLIP model for text and image prompt fusion")
    parser.add_argument("--clip_variant", type=str, default="biomedclip", choices=["biomedclip", "clip", "bioclip"], help="Which CLIP variant to use")
    parser.add_argument("--tokenizer", type=str, default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", help="Which tokenizer to use")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to the prompt file containing organ-specific text prompts")

    args = parser.parse_args()
    print("Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared.")
    join = os.path.join
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    use_clip_str = "_use_clip" if args.use_clip else "_no_clip"
    clip_variant = args.clip_variant
    tokenizer_name = args.tokenizer
    
    # Create prompt_id to distinguish different prompt configurations
    prompt_id = "no_prompt" if args.prompt_file is None else os.path.splitext(os.path.basename(args.prompt_file))[0]

    all_organs = list(ORGANS.keys())
    seed = 2031
    random.seed(seed)
    train_organs = random.sample(all_organs, 8)
    test_organs = [o for o in all_organs if o not in train_organs]

    print("Train organs:", [ORGANS[o] for o in train_organs])
    print("Test  organs:", [ORGANS[o] for o in test_organs])


    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "organ_split.json"), "w") as f:
        json.dump({"seed": seed, "train": train_organs, "test": test_organs}, f)

    model_save_path = join(
        args.work_dir,
        f"{args.task_name}_{prompt_id}_MS{args.ms_features}_oneneck{args.one_neck}_{args.clip_variant}_{run_id}"
    )

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
        clip_variant=args.clip_variant,
    )

    if args.clip_variant == "biomedclip":
        tokenizer_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif args.clip_variant == "clip":
        tokenizer_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif args.clip_variant == "bioclip":
        tokenizer_name = "imageomics/bioclip"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Unknown clip_variant: {args.clip_variant}")
    print(f"[INFO] clip_variant: {args.clip_variant}, tokenizer: {tokenizer_name}")

    train_loaders, val_loaders, test_loaders = build_loaders(
        args.tr_npy_path, tokenizer,
        train_organs=train_organs, test_organs=test_organs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        prompt_file=args.prompt_file)


    if torch.cuda.is_available() and args.device.startswith("cuda"):
        medsam_model = medsam_model.to(args.device)
    medsam_model.train()

    optimizer = torch.optim.AdamW(
        medsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    ce_loss = nn.BCEWithLogitsLoss()

    checkpoint_path = args.resume if args.resume else args.checkpoint
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

    val_losses = {ORGANS[oid]: [] for oid in val_loaders.keys()}
    val_accuracies = {ORGANS[oid]: [] for oid in val_loaders.keys()}
    # num_epochs=100
    best_loss = 1e10
    iter_num=0
    if args.use_wandb:
        print("Using wandb")
        wandb.init(project="SAM")
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0
        epoch_dice = 0
        for oid, train_loader in train_loaders.items():
            from tqdm import tqdm
            for image, gt2D, boxes, _, text_input in tqdm(train_loader, desc=f"Training {ORGANS[oid]}", leave=False):
                optimizer.zero_grad()
                image, gt2D = image.to(device), gt2D.to(device)
                boxes_np = boxes.numpy()
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = medsam_model(image, boxes_np, text_input)
                        loss = loss_fn(pred, gt2D.float())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    pred = medsam_model(image, boxes_np, text_input)
                    loss = loss_fn(pred, gt2D.float())
                    loss.backward();
                    optimizer.step()

                dice = compute_dice_coefficient(gt2D.cpu().numpy(),
                                                (pred > 0.5).cpu().numpy())
                epoch_loss += loss.item()
                epoch_dice += dice

                iter_num += 1

        total_batches = sum(len(dl) for dl in train_loaders.values())
        epoch_loss /= total_batches
        epoch_dice /= total_batches
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_dice)

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss, "epoch_dice": epoch_dice})
            
        
        # Validation loop
        medsam_model.eval()
        val_results = defaultdict(dict)
        with torch.no_grad():

            for oid, val_loader in val_loaders.items():
                organ = ORGANS[oid]

                total_loss, total_dice, count = 0, 0, 0

                # use tqdm to show progress bar
                for image, gt, boxes, _, text_input in tqdm(val_loader, desc=f"Validating {organ}", leave=False):
                    bboxes = boxes.numpy()
                    image, gt = image.to(device), gt.to(device)

                    # model prediction
                    medsam_pred = medsam_model(image, bboxes, text_input)
                    loss = seg_loss(medsam_pred, gt.float()) + ce_loss(medsam_pred, gt.float())
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

            # for test
            if (epoch + 1) % 10 == 0 or epoch + 1 == args.num_epochs:
                medsam_model.eval()
                with torch.no_grad():
                    for oid, dl in test_loaders.items():
                        organ = ORGANS[oid]
                        total_loss, total_dice, cnt = 0, 0, 0
                        for img, gt, boxes, _, text in tqdm(dl, desc=f"Testing {organ}", leave=False):
                            pred = medsam_model(img.to(device),
                                                boxes.cpu().numpy(),
                                                text.to(device))
                            loss = seg_loss(pred, gt.to(device).float()) + \
                                   ce_loss(pred, gt.to(device).float())
                            dice = compute_dice_coefficient(gt.numpy(),
                                                            (pred > 0.5).cpu().numpy())
                            total_loss += loss.item();
                            total_dice += dice;
                            cnt += 1
                        print(f"[TEST] {organ}: Loss={total_loss / cnt:.4f}, Dice={total_dice / cnt:.4f}")

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
        
        # Save with error handling and retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Save to temporary file first
                temp_path = join(model_save_path, f"medsam_model_latest_temp_{attempt}.pth")
                torch.save(checkpoint, temp_path)
                
                # Verify the saved file
                test_load = torch.load(temp_path, map_location='cpu')
                if 'model' in test_load and 'optimizer' in test_load:
                    # Atomic move to final location
                    final_path = join(model_save_path, "medsam_model_latest.pth")
                    os.rename(temp_path, final_path)
                    print(f"Latest model saved successfully to {final_path}")
                    break
                else:
                    raise ValueError("Checkpoint verification failed")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    print(f"Failed to save latest model after {max_retries} attempts")
                else:
                    import time
                    time.sleep(2)  # Wait before retry

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
            
            # Save with error handling and retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Save to temporary file first
                    temp_path = join(model_save_path, f"medsam_model_best_temp_{attempt}.pth")
                    torch.save(checkpoint, temp_path)
                    
                    # Verify the saved file
                    test_load = torch.load(temp_path, map_location='cpu')
                    if 'model' in test_load and 'optimizer' in test_load:
                        # Atomic move to final location
                        final_path = join(model_save_path, "medsam_model_best.pth")
                        os.rename(temp_path, final_path)
                        print("saved a better ckpt.")
                        break
                    else:
                        raise ValueError("Checkpoint verification failed")
                        
                except Exception as e:
                    print(f"Best model save attempt {attempt + 1} failed: {e}")
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    if attempt == max_retries - 1:
                        print(f"Failed to save best model after {max_retries} attempts")
                    else:
                        import time
                        time.sleep(2)  # Wait before retry

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
