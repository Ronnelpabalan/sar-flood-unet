"""PyTorch Dataset for the Sen1Floods11 hand-labelled Sentinel-1 chips.

Each item is (image, label):
    image: float32 tensor (2, 512, 512), bands [VV, VH] clipped and scaled to 0-1
    label: int64 tensor (512, 512), 1 = water, 0 = not water, -1 = ignore
Pixels with no label or missing SAR (NaN) are set to -1, so the loss and the
metrics skip exactly the same pixels as the threshold baselines.
"""

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# Clip ranges in dB: 0.5th and 99.5th percentiles of valid train pixels
# (notebooks/01_explore_data.ipynb, Cell 10).
CLIP_DB = {"VV": (-25.7, -0.8), "VH": (-38.1, -8.4)}


def read_split(data_dir, split):
    """Return the chip IDs (e.g. 'Ghana_103272') of an official split."""
    df = pd.read_csv(f"{data_dir}/splits/flood_{split}_data.csv",
                     header=None, names=["s1", "label"])
    return [f.replace("_S1Hand.tif", "") for f in df["s1"]]


def load_chip(data_dir, cid):
    """Return (s1, label): s1 is (2, H, W) float32 [VV, VH] in dB, label is (H, W) int16."""
    with rasterio.open(f"{data_dir}/S1Hand/{cid}_S1Hand.tif") as src:
        s1 = src.read().astype("float32")
    with rasterio.open(f"{data_dir}/LabelHand/{cid}_LabelHand.tif") as src:
        label = src.read(1)
    return s1, label


def normalise(s1):
    """Clip each band to CLIP_DB, scale to 0-1, and replace NaN with 0."""
    out = np.empty_like(s1, dtype="float32")
    for i, (lo, hi) in enumerate([CLIP_DB["VV"], CLIP_DB["VH"]]):
        out[i] = (np.clip(s1[i], lo, hi) - lo) / (hi - lo)
    return np.nan_to_num(out, nan=0.0)


def augment(image, label, rng):
    """Random flips and 90-degree rotations, applied identically to image and label."""
    if rng.random() < 0.5:
        image, label = image[:, :, ::-1], label[:, ::-1]      # left-right flip
    if rng.random() < 0.5:
        image, label = image[:, ::-1, :], label[::-1, :]      # up-down flip
    k = int(rng.integers(4))                                   # rotate 0/90/180/270 degrees
    image, label = np.rot90(image, k, axes=(1, 2)), np.rot90(label, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class Sen1Floods11(Dataset):
    """Hand-labelled Sen1Floods11 chips for one split, preloaded into memory.

    data_dir: folder containing S1Hand/, LabelHand/ and splits/
    split:    'train', 'valid', 'test' or 'bolivia'
    augment:  apply random flips/rotations (use for training only)
    """

    def __init__(self, data_dir, split, augment=False, seed=0):
        self.chip_ids = read_split(data_dir, split)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.images, self.labels = [], []
        for cid in self.chip_ids:
            s1, label = load_chip(data_dir, cid)
            label = label.astype("int64")
            label[np.isnan(s1).any(axis=0)] = -1               # missing SAR -> ignore
            self.images.append(normalise(s1))
            self.labels.append(label)

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.augment:
            image, label = augment(image, label, self.rng)
        return torch.from_numpy(image), torch.from_numpy(label)
