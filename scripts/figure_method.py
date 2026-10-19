#!/usr/bin/env python
"""Method illustration: image | log-amplitude spectrum | spectral-residual saliency (64, 128 px).

python scripts/figure_method.py --splits data/splits.csv --per-class 1
Writes paper/figures/spectral_examples.pdf. Uses the *untrained* (fixed Hou & Zhang) residual,
so the figure shows the prior that SpecTexNet starts from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fabricnet.data import build_transforms  # noqa: E402
from fabricnet.models import SpectralResidual  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--out", default="paper/figures/spectral_examples")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    df = pd.read_csv(a.splits)
    df = df[df["kind"].ne("derived")] if "kind" in df else df
    df["path"] = [p if Path(p).is_absolute() else str(ROOT / p) for p in df["path"]]
    rows = pd.concat([g.sample(min(a.per_class, len(g)), random_state=a.seed) for _, g in df.groupby("label")])
    sr = SpectralResidual(scales=(64, 128), learnable=False, out_size=224).eval()
    tf = build_transforms(224, train=False)
    cols = ["Image", "log |FFT| (centred)", "SR saliency 64", "SR saliency 128"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(r["path"]).convert("L").convert("RGB").resize((224, 224))
        x = tf(img)[None]
        g = np.asarray(img.convert("L"), dtype=np.float32)
        spec = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) + 1)
        with torch.no_grad():
            s = sr(x)[0].numpy()
        for j, p in enumerate([np.asarray(img), spec, s[0], s[1]]):
            axes[i, j].imshow(p, cmap=None if j == 0 else ("gray" if j == 1 else "inferno"))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=8)
        axes[i, 0].set_ylabel(r["label"].replace("_", " "), fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=300)
    fig.savefig(a.out + ".png", dpi=150)
    print(f"saved {a.out}.pdf")


if __name__ == "__main__":
    main()
