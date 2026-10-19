from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .transforms import GeometricAug, build_transforms


class FabricDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 224, train: bool = False,
                 geo_mode: str = "oa", strength: str = "medium", classes=None, cache: bool = False,
                 grayscale: bool = False):
        self.paths = df["path"].tolist()
        self.ys = df["y"].astype(int).tolist()
        self.train = train
        kw = {"classes": classes} if classes else {}
        self.geo = GeometricAug(geo_mode, **kw) if train else None
        self.tf = build_transforms(img_size, train, strength)
        self._cache = {} if cache else None
        self.grayscale = grayscale

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i].copy()
        img = Image.open(self.paths[i])
        img = img.convert("L").convert("RGB") if self.grayscale else img.convert("RGB")
        if self._cache is not None:
            self._cache[i] = img.copy()
        return img

    def __getitem__(self, i):
        img, y = self._load(i), self.ys[i]
        if self.geo is not None:
            img, y = self.geo(img, y)
        return self.tf(img), torch.tensor(y, dtype=torch.long), i


def balanced_sampler(ys: list[int], power: float = 0.5) -> WeightedRandomSampler:
    """Sampler with weight ~ 1/count^power (power=1 fully balanced, 0 = natural)."""
    counts = torch.bincount(torch.tensor(ys))
    w = (1.0 / counts.float().clamp(min=1) ** power)[torch.tensor(ys)]
    return WeightedRandomSampler(w.double(), num_samples=len(ys), replacement=True)
