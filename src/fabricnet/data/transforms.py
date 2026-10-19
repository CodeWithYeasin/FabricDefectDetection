"""Augmentations, including Orientation-Aware rotation (OA-Rot).

Some defect classes are defined by direction: a ``horizontal`` defect rotated by 90 degrees
*is* a ``vertical`` defect. Ordinary rotation augmentation therefore creates wrong labels,
while dropping rotation throws away a cheap 4x data multiplier for every other class.
OA-Rot rotates by k*90 degrees and, for odd k, swaps the orientation-paired labels.
Flips are always label-preserving (a horizontal line stays horizontal under either flip).

Modes (``geo_mode``):
  none       no geometric augmentation
  flip       h/v flips only (label-safe baseline)
  naive_rot  flips + rot90 *without* label swap (ablation: shows the cost of label noise)
  oa         flips + rot90 with label swap (proposed)
"""
from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from .. import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def orientation_swap_map(classes: list[str] = CLASSES, pairs=(("horizontal", "vertical"),)) -> list[int]:
    """Return permutation ``perm`` with perm[y] = label after a 90-degree rotation."""
    perm = list(range(len(classes)))
    for a, b in pairs:
        if a in classes and b in classes:
            ia, ib = classes.index(a), classes.index(b)
            perm[ia], perm[ib] = ib, ia
    return perm


class GeometricAug:
    def __init__(self, mode: str = "oa", classes: list[str] = CLASSES, p_flip: float = 0.5):
        assert mode in {"none", "flip", "naive_rot", "oa"}, mode
        self.mode, self.p_flip = mode, p_flip
        self.perm = orientation_swap_map(classes)

    def __call__(self, img: Image.Image, y: int) -> tuple[Image.Image, int]:
        if self.mode == "none":
            return img, y
        if random.random() < self.p_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.p_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.mode in {"naive_rot", "oa"}:
            k = random.randint(0, 3)
            if k:
                img = img.transpose([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k])
                if self.mode == "oa" and k % 2 == 1:
                    y = self.perm[y]
        return img, y


def build_transforms(img_size: int = 224, train: bool = True, strength: str = "medium"):
    """Photometric + crop transforms applied after the geometric step (label-independent)."""
    if not train:
        return T.Compose(
            [
                T.ToImage(),
                T.Resize((img_size, img_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # Textures: keep crops large so small defects (holes) are not cropped out.
    scale = {"light": (0.85, 1.0), "medium": (0.7, 1.0), "strong": (0.5, 1.0)}[strength]
    jitter = {"light": 0.1, "medium": 0.2, "strong": 0.3}[strength]
    return T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop((img_size, img_size), scale=scale, ratio=(0.9, 1.1), antialias=True),
            T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter / 2, hue=0.02),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
