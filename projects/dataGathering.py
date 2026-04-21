"""
JHU-CROWD++ v2.0 — data loading for crowd counting (PyTorch density maps)
and crowd tracking (bounding boxes).

Dataset structure expected under `data_root/`:
  {train,val,test}/
    images/       *.jpg
    gt/           *.txt   — "x y w h occlusion blur" per head
    image_labels.txt      — "id,count,scene,weather,distractor"
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JHUCrowdDataset(Dataset):
    """
    Two modes:
      generate_density=True  → returns density maps (crowd counting)
      generate_density=False → returns bounding boxes  (tracking / detection)
    """

    WEATHER = {0: "no-degradation", 1: "fog/haze", 2: "rain", 3: "snow"}
    OCCLUSION = {1: "visible", 2: "partial", 3: "full"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: tuple | None = None,
        augment: bool = False,
        generate_density: bool = True,
        density_sigma: float = 15.0,
    ):
        """
        Args:
            root_dir:         path to the `data/` folder
            split:            "train", "val" or "test"
            img_size:         (H, W) to resize images, or None to keep originals
            augment:          random horizontal flip + colour jitter (training only)
            generate_density: True  → counting mode, False → tracking mode
            density_sigma:    Gaussian σ for density-map generation
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.generate_density = generate_density
        self.density_sigma = density_sigma

        self.images_dir = os.path.join(root_dir, split, "images")
        self.gt_dir = os.path.join(root_dir, split, "gt")
        self.labels_file = os.path.join(root_dir, split, "image_labels.txt")

        self.image_labels = self._load_image_labels()
        self.image_ids = list(self.image_labels.keys())

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _load_image_labels(self) -> dict:
        labels = {}
        with open(self.labels_file, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 5:
                    continue
                labels[parts[0]] = {
                    "count": int(parts[1]),
                    "scene": parts[2],
                    "weather": int(parts[3]),
                    "distractor": int(parts[4]),
                }
        return labels

    def _load_annotations(self, img_id: str) -> list[dict]:
        gt_path = os.path.join(self.gt_dir, f"{img_id}.txt")
        anns = []
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        anns.append(
                            {
                                "x": int(parts[0]),
                                "y": int(parts[1]),
                                "w": int(parts[2]),
                                "h": int(parts[3]),
                                "occlusion": int(parts[4]),
                                "blur": int(parts[5]),
                            }
                        )
        return anns

    # ------------------------------------------------------------------
    # Density-map generation
    # ------------------------------------------------------------------

    def _make_density_map(
        self, orig_h: int, orig_w: int, anns: list[dict], out_h: int, out_w: int
    ) -> np.ndarray:
        """
        Place a Gaussian at each head centre, then resize to (out_h, out_w).
        The integral of the map equals the head count.
        """
        dm = np.zeros((orig_h, orig_w), dtype=np.float32)
        for a in anns:
            x, y = a["x"], a["y"]
            if 0 <= x < orig_w and 0 <= y < orig_h:
                dm[y, x] += 1.0
        dm = gaussian_filter(dm, sigma=self.density_sigma)

        if (out_h, out_w) != (orig_h, orig_w):
            # Resize and restore the integral (total count)
            total = dm.sum()
            dm_pil = Image.fromarray(dm).resize((out_w, out_h), Image.BILINEAR)
            dm = np.array(dm_pil, dtype=np.float32)
            if dm.sum() > 0:
                dm *= total / dm.sum()

        return dm

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _apply_augmentation(self, image: Image.Image, anns: list[dict]) -> tuple:
        """Random horizontal flip applied consistently to image and annotations."""
        if np.random.rand() < 0.5:
            w = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            anns = [{**a, "x": w - 1 - a["x"]} for a in anns]
        return image, anns

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        label_info = self.image_labels[img_id]

        # Load image
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load annotations
        anns = self._load_annotations(img_id)

        # Augmentation (train only)
        if self.augment:
            image, anns = self._apply_augmentation(image, anns)

        # Resize image
        if self.img_size is not None:
            out_h, out_w = self.img_size
            image = image.resize((out_w, out_h), Image.BILINEAR)
        else:
            out_h, out_w = orig_h, orig_w

        # Colour jitter (train only)
        if self.augment:
            image = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )(image)

        # → tensor + normalise
        img_tensor = self.normalize(transforms.ToTensor()(image))

        # ---- Counting mode: density map --------------------------------
        if self.generate_density:
            dm = self._make_density_map(orig_h, orig_w, anns, out_h, out_w)
            density_tensor = torch.from_numpy(dm).unsqueeze(0)  # (1, H, W)
            return {
                "image": img_tensor,          # (3, H, W)
                "density_map": density_tensor,  # (1, H, W)  — sum ≈ count
                "count": label_info["count"],
                "image_id": img_id,
            }

        # ---- Tracking mode: bounding boxes -----------------------------
        sx = out_w / orig_w
        sy = out_h / orig_h
        if anns:
            boxes = torch.tensor(
                [
                    [
                        (a["x"] - a["w"] / 2) * sx,
                        (a["y"] - a["h"] / 2) * sy,
                        (a["x"] + a["w"] / 2) * sx,
                        (a["y"] + a["h"] / 2) * sy,
                    ]
                    for a in anns
                ],
                dtype=torch.float32,
            )
            occlusion = torch.tensor([a["occlusion"] for a in anns], dtype=torch.long)
            blur = torch.tensor([a["blur"] for a in anns], dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            occlusion = torch.zeros(0, dtype=torch.long)
            blur = torch.zeros(0, dtype=torch.long)

        return {
            "image": img_tensor,            # (3, H, W)
            "boxes": boxes,                 # (N, 4)  xyxy format
            "occlusion": occlusion,         # (N,)  1=visible 2=partial 3=full
            "blur": blur,                   # (N,)  0=sharp 1=blurred
            "count": label_info["count"],
            "scene": label_info["scene"],
            "weather": label_info["weather"],
            "distractor": label_info["distractor"],
            "image_id": img_id,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _collate_tracking(batch: list[dict]) -> dict:
    """Variable-length annotations need a custom collate (tracking mode)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "occlusion": [b["occlusion"] for b in batch],
        "blur": [b["blur"] for b in batch],
        "count": torch.tensor([b["count"] for b in batch]),
        "scene": [b["scene"] for b in batch],
        "weather": [b["weather"] for b in batch],
        "distractor": [b["distractor"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    img_size: tuple | None = None,
    generate_density: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    shuffle: bool | None = None,
) -> tuple[DataLoader, JHUCrowdDataset]:
    """
    Returns (DataLoader, Dataset) for the requested split.

    num_workers=0 is the safest default on Windows.
    Set augment=True only for training splits.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = JHUCrowdDataset(
        root_dir=data_root,
        split=split,
        img_size=img_size,
        augment=augment,
        generate_density=generate_density,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=None if generate_density else _collate_tracking,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Quick statistics
# ---------------------------------------------------------------------------

def print_split_stats(data_root: str, split: str) -> None:
    labels_file = os.path.join(data_root, split, "image_labels.txt")
    counts, scenes, weathers = [], {}, {0: 0, 1: 0, 2: 0, 3: 0}

    with open(labels_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 5:
                continue
            counts.append(int(parts[1]))
            scenes[parts[2]] = scenes.get(parts[2], 0) + 1
            weathers[int(parts[3])] += 1

    weather_labels = {0: "clear", 1: "fog/haze", 2: "rain", 3: "snow"}
    print(f"\n=== {split.upper()} ({len(counts)} images) ===")
    print(f"  Count  — min {min(counts):,}  max {max(counts):,}  "
          f"mean {np.mean(counts):.1f}  median {np.median(counts):.1f}")
    print(f"  Scenes — {dict(sorted(scenes.items(), key=lambda kv: -kv[1]))}")
    print(f"  Weather— {dict((weather_labels[k], v) for k, v in weathers.items())}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Data root: {DATA_ROOT}")

    for split in ("train", "val", "test"):
        print_split_stats(DATA_ROOT, split)

    # --- counting mode ---
    print("\n--- Counting mode (density map) ---")
    ds = JHUCrowdDataset(DATA_ROOT, split="train", img_size=(512, 512),
                         generate_density=True, augment=False)
    s = ds[0]
    print(f"  image      : {s['image'].shape}")
    print(f"  density_map: {s['density_map'].shape}  "
          f"sum={s['density_map'].sum():.1f}  gt={s['count']}")

    # --- tracking mode ---
    print("\n--- Tracking mode (bounding boxes) ---")
    ds_t = JHUCrowdDataset(DATA_ROOT, split="train", img_size=(512, 512),
                            generate_density=False, augment=False)
    s_t = ds_t[0]
    print(f"  image  : {s_t['image'].shape}")
    print(f"  boxes  : {s_t['boxes'].shape}  (N heads × [x1,y1,x2,y2])")
    print(f"  count={s_t['count']}  scene={s_t['scene']}  "
          f"weather={JHUCrowdDataset.WEATHER[s_t['weather']]}")

    # --- DataLoader ---
    print("\n--- DataLoader (batch_size=4, counting) ---")
    loader, _ = get_dataloader(DATA_ROOT, split="train", batch_size=4,
                                img_size=(512, 512), generate_density=True,
                                augment=True, num_workers=0)
    batch = next(iter(loader))
    print(f"  images      : {batch['image'].shape}")
    print(f"  density_maps: {batch['density_map'].shape}")
    print(f"  counts      : {batch['count'].tolist()}")