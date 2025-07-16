import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import monai
from collections import defaultdict
from segment_anything import sam_model_registry
from train_one_gpu import NpyDataset, MedSAM, train_val_split
from utils.SurfaceDice import (
    compute_dice_coefficient,
    compute_surface_distances,
    compute_surface_dice_at_tolerance
)

def compute_ci(data_list, alpha=0.95, n_samples=10, subsample_size=50, seed=1):
    np.random.seed(seed)
    metrics = []
    if len(data_list) < subsample_size:
        subsample_size = len(data_list)
    for _ in range(n_samples):
        subsample = np.random.choice(data_list, size=subsample_size, replace=False)
        metrics.append(np.mean(subsample))
    metrics = np.sort(metrics)
    alpha_lower = (1 - alpha) / 2
    alpha_upper = (1 + alpha) / 2
    lower_idx = int(alpha_lower * n_samples)
    upper_idx = int(alpha_upper * n_samples)
    ci_lower = metrics[lower_idx]
    ci_upper = metrics[upper_idx]
    mean_val = np.mean(metrics)
    return mean_val, (ci_lower, ci_upper)


def compute_loss(pred_mask, gt):
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    loss = dice_loss_fn(pred_mask, gt) + bce_loss_fn(pred_mask, gt.float())
    return loss


def evaluate_organ(medsam_model, data_loader, device, pixel_spacing_2d, tolerance):
    organ_dsc_list = []
    organ_nsd_list = []
    organ_hd_list = []

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for step, (image, gt, boxes, img_name, text_input) in enumerate(tqdm(data_loader)):
            boxes_np = boxes.cpu().numpy()
            image, gt = image.to(device), gt.to(device)

            pred_mask = medsam_model(image, boxes_np, text_input)
            loss = compute_loss(pred_mask, gt)
            total_loss += loss.item()
            count += 1

            dsc_val = compute_dice_coefficient(gt.cpu().numpy(), (pred_mask > 0.5).cpu().numpy())
            organ_dsc_list.append(dsc_val)

            gt_np = gt.cpu().numpy()
            pred_np = (pred_mask > 0.5).cpu().numpy()

            spacing_3d = [1.0, pixel_spacing_2d[0], pixel_spacing_2d[1]]

            for b in range(pred_np.shape[0]):
                gt_3d = gt_np[b, 0].astype(bool)[None, ...]
                pred_3d = pred_np[b, 0].astype(bool)[None, ...]

                surface_info = compute_surface_distances(mask_gt=gt_3d, mask_pred=pred_3d, spacing_mm=spacing_3d)

                nsd_val = compute_surface_dice_at_tolerance(surface_info, tolerance)
                organ_nsd_list.append(nsd_val)

                dist_gt_to_pred = surface_info["distances_gt_to_pred"]
                dist_pred_to_gt = surface_info["distances_pred_to_gt"]
                hd_val = max(dist_gt_to_pred.max(), dist_pred_to_gt.max()) if dist_gt_to_pred.size > 0 and dist_pred_to_gt.size > 0 else np.inf
                organ_hd_list.append(hd_val)

    return total_loss, count, organ_dsc_list, organ_nsd_list, organ_hd_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/flare2021npy", help="Root directory of dataset.")
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
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
    parser.add_argument("--val_organs", nargs='+', type=str, default=["liver", "kidney", "spleen", "pancreas"],
                        help="List of organs for validation or testing.")
    ### new params
    parser.add_argument("--ms_features", action="store_true")
    parser.add_argument("--one_neck", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    parser.add_argument("--pixel_spacing", nargs='+', type=float, default=[1.0, 1.0],
                        help="Pixel spacing for Surface Dice.")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Tolerance for surface dice.")
    parser.add_argument("--alpha", type=float, default=0.95, help="Confidence level for CI.")
    parser.add_argument("--use_clip", type=bool, default=True,help="Whether to use CLIP model for text and image prompt fusion")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    full_dataset = NpyDataset(args.data_root)
    _, val_datasets = train_val_split(full_dataset, val_organs=args.val_organs)

    sam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM-ViT-B_MSTrue_oneneckFalse_use_clip_20250215-2152/medsam_model_best.pth",
        ms_features=args.ms_features,
        one_neck=args.one_neck,)
    medsam_model = MedSAM(image_encoder=sam_model.image_encoder, mask_decoder=sam_model.mask_decoder, prompt_encoder=sam_model.prompt_encoder).to(device)

    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    medsam_model.load_state_dict(checkpoint_data["model"], strict=True)
    medsam_model.eval()

    organ_results = defaultdict(dict)

    for organ, val_subset in val_datasets.items():
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        total_loss, count, organ_dsc_list, organ_nsd_list, organ_hd_list = evaluate_organ(
            medsam_model=medsam_model, data_loader=val_loader, device=device, pixel_spacing_2d=args.pixel_spacing, tolerance=args.tolerance
        )

        avg_loss = total_loss / count
        dsc_mean, dsc_ci = compute_ci(organ_dsc_list, alpha=args.alpha)
        nsd_mean, nsd_ci = compute_ci(organ_nsd_list, alpha=args.alpha)
        hd_mean, hd_ci = compute_ci(organ_hd_list, alpha=args.alpha)

        organ_results[organ]["avg_loss"] = avg_loss
        organ_results[organ]["dsc_mean"] = dsc_mean
        organ_results[organ]["dsc_ci"] = dsc_ci
        organ_results[organ]["nsd_mean"] = nsd_mean
        organ_results[organ]["nsd_ci"] = nsd_ci
        organ_results[organ]["hd_mean"] = hd_mean
        organ_results[organ]["hd_ci"] = hd_ci

        print(f"[{organ}] Samples: {count}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  DSC : {dsc_mean:.4f} (95% CI: [{dsc_ci[0]:.4f}, {dsc_ci[1]:.4f}])")
        print(f"  NSD : {nsd_mean:.4f} (95% CI: [{nsd_ci[0]:.4f}, {nsd_ci[1]:.4f}])")
        print(f"  HD  : {hd_mean:.4f} (95% CI: [{hd_ci[0]:.4f}, {hd_ci[1]:.4f}])")


if __name__ == "__main__":
    main()
