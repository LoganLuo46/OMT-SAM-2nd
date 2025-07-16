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
from torch.utils.data import get_worker_info

class OrganSliceDataset(Dataset):
    """
    针对单一器官 (organ_id) 的切片级数据集：
        imgs → 3×1024×1024  float32  [0,1]
        gt   → 1×256×256    int64    {0,1}
        bbox → 4            float32  [x1,y1,x2,y2]
    """
    def __init__(self, npz_dir, organ_id, tokenizer=None,
                 clip_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 bbox_shift=20, img_size=1024, gt_size=256,
                 prompt_file="organ_prompts_ten_2nd_version.txt",

                 ):
        super().__init__()
        self.npz_files  = sorted(glob.glob(join(npz_dir, "*.npz")))
        self.organ_id   = organ_id
        self.organ_name = ORGANS[organ_id]
        self.bbox_shift = bbox_shift
        self.img_size   = img_size
        self.gt_size    = gt_size

        # --- 读取 prompt 文件 ---
        self.prompt_bank = {}
        with open(prompt_file, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    k = k.strip().lower()
                    self.prompt_bank.setdefault(k, []).append(v.strip())

        # Use passed tokenizer if provided, otherwise create one
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if clip_model_name.startswith("hf-hub:"):
                tokenizer_repo = clip_model_name.split("hf-hub:", 1)[1]  # →  imageomics/bioclip
            else:
                tokenizer_repo = clip_model_name  # 直接是 repo id
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

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



        prompt_pool = self.prompt_bank.get(self.organ_name.lower(), [self.organ_name])

        # ① 取得当前 worker 信息
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程 DataLoader，直接用全局 random
            prompt_text = random.choice(prompt_pool)
        else:
            # ② 为每个 worker 创建独立的 numpy RNG，seed = base_seed + worker_id
            #    worker_info.seed 是 PyTorch 自动分配的基础种子
            rng = np.random.default_rng(worker_info.seed)
            prompt_text = rng.choice(prompt_pool)


        text_ids = self.tokenizer(
            prompt_text,
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

