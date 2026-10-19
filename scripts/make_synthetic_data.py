#!/usr/bin/env python
"""Generate a tiny synthetic woven-texture dataset with the six classes.

Used ONLY for smoke tests / CI and for sanity-checking that the spectral-residual branch
lights up on periodicity breaks. Never report numbers from this data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

FOLDERS = {"defect_free": "defect free", "hole": "hole", "horizontal": "horizontal",
           "lines": "lines", "stain": "stain", "vertical": "Vertical"}


def weave(rng, n=256):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    p1, p2 = rng.uniform(5, 9), rng.uniform(5, 9)
    th = rng.uniform(-0.05, 0.05)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    t = 0.5 + 0.25 * np.sin(2 * np.pi * u / p1) * np.sin(2 * np.pi * v / p2) + 0.1 * np.sin(2 * np.pi * u / p1)
    t += rng.normal(0, 0.03, t.shape)
    base = rng.uniform(0.3, 0.8, 3)
    return np.clip(t[..., None] * base[None, None] * 1.6, 0, 1)


def add_defect(img, cls, rng):
    n = img.shape[0]
    cy, cx = rng.integers(n // 5, 4 * n // 5, 2)
    yy, xx = np.mgrid[0:n, 0:n]
    if cls == "hole":
        r = rng.integers(6, 14)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = rng.uniform(0, 0.15)
    elif cls == "stain":
        r = rng.integers(15, 35)
        m = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))[..., None]
        img = img * (1 - 0.6 * m) + 0.6 * m * np.array([0.45, 0.3, 0.1])
    elif cls == "horizontal":
        w = rng.integers(2, 5)
        img[cy : cy + w, :] *= 0.4
    elif cls == "vertical":
        w = rng.integers(2, 5)
        img[:, cx : cx + w] *= 0.4
    elif cls == "lines":
        # oblique scratch: not horizontal or vertical, so rotation keeps the class
        ang = rng.choice([np.pi / 4, 3 * np.pi / 4]) + rng.uniform(-0.2, 0.2)
        d = np.abs((xx - cx) * np.sin(ang) - (yy - cy) * np.cos(ang))
        img[d < 1.8] = 0.95
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls, folder in FOLDERS.items():
        d = Path(a.out) / folder
        d.mkdir(parents=True, exist_ok=True)
        n = a.per_class * (3 if cls == "defect_free" else 1)  # mimic imbalance
        for i in range(n):
            img = weave(rng)
            if cls != "defect_free":
                img = add_defect(img, cls, rng)
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{cls}_{i:04d}.jpg", quality=92)
    print(f"wrote synthetic data to {a.out}")


if __name__ == "__main__":
    main()
