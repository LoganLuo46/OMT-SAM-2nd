
# id ↔ name 按你截图确定
ORGANS = {
     1: "liver",
     2: "right_kidney",
     3: "spleen",
     4: "pancreas",
     5: "aorta",
     6: "ivc",
     7: "right_adrenal_gland",
     8: "left_adrenal_gland",
     9: "gallbladder",
    10: "esophagus",
    11: "stomach",
    13: "left_kidney",
}


import os, glob, random, numpy as np, torch
from os.path import join, basename
from skimage.transform import resize
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class OrganSliceDataset(Dataset):
    """
    针对单一器官 (organ_id) 的切片级数据集：
        imgs → 3×1024×1024  float32  [0,1]
        gt   → 1×256×256    int64    {0,1}
        bbox → 4            float32  [x1,y1,x2,y2]
    """
    def __init__(self, npz_dir, organ_id,
                 bbox_shift=20, img_size=1024, gt_size=256,
                 ):
        super().__init__()
        self.npz_files  = sorted(glob.glob(join(npz_dir, "*.npz")))
        self.organ_id   = organ_id
        self.organ_name = ORGANS[organ_id]
        self.bbox_shift = bbox_shift
        self.img_size   = img_size
        self.gt_size    = gt_size
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.slice_map = []
        for fid, f in enumerate(self.npz_files):
            with np.load(f) as d:
                gts = d["gts"]
            for sid, slice_gt in enumerate(gts):
                if np.any(slice_gt == organ_id):
                    self.slice_map.append((fid, sid))

        if not self.slice_map:
            raise RuntimeError(f"No slice with organ id {organ_id} found.")

        print(f"[{ORGANS[organ_id]}]  cases: {len(self.npz_files)}, "
              f"slices: {len(self.slice_map)}")

    @staticmethod
    def _resize(arr, tgt, order):
        return resize(arr, (tgt, tgt), order=order,
                      preserve_range=True, mode="constant",
                      anti_aliasing=(order != 0)).astype(arr.dtype)

    def __len__(self): return len(self.slice_map)

    def __getitem__(self, idx):
        fid, sid = self.slice_map[idx]
        npz_path = self.npz_files[fid]
        case     = os.path.splitext(basename(npz_path))[0]

        with np.load(npz_path) as d:
            img = d["imgs"][sid]      # (H,W) uint8
            gt  = d["gts"][sid]       # (H,W) uint8

        # resize
        if img.shape[0] != self.img_size:
            img = self._resize(img, self.img_size, order=3)
        if gt.shape[0] != self.gt_size:
            gt  = self._resize(gt,  self.gt_size, order=0)

        img = np.repeat(img[:, :, None], 3, axis=-1).astype("float32") / 255.
        img = np.transpose(img, (2,0,1))            # (3,H,W)
        gt = self._resize(gt, self.img_size, order=0)

        gt2D = (gt == self.organ_id).astype("uint8")# (H,W)

        # bbox
        y_idx, x_idx = np.where(gt2D > 0)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bbox  = np.array([x_min, y_min, x_max, y_max], dtype="float32")


        text_ids = self.tokenizer(
            self.organ_name,
            truncation=True, padding="max_length",
            max_length=77, return_tensors="pt"
        )["input_ids"].squeeze(0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(gt2D[None], dtype=torch.long),
            torch.tensor(bbox, dtype=torch.float32),
            f"{case}-{sid:03d}.npy",
            text_ids
        )
import torch, random, numpy as np
from torch.utils.data import DataLoader

def build_dataloaders():
    root_dir = "data/npy/CT_Abd"
    batch_size, num_workers = 0, 0
    dataloaders = {}
    for oid, name in ORGANS.items():
        try:
            ds = OrganSliceDataset(root_dir, organ_id=oid)
            dl = DataLoader(ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
            dataloaders[oid] = dl
        except RuntimeError as e:
            print(f"[{name}] skipped:", e)
    return dataloaders

def main():
    dls = build_dataloaders()
    imgs, gts, bboxes, names = next(iter(dls[4]))   # 4 = pancreas
    print(imgs.shape, gts.shape, names[:3])

if __name__ == "__main__":
    main()
