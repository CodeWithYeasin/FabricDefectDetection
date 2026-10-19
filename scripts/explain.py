#!/usr/bin/env python
"""Qualitative figure: for N test images per class -> input | Grad-CAM | spectral saliency | SGA gate.

python scripts/explain.py --exp spectex_convnext_tiny --fold 0 --per-class 2
Writes paper/figures/explain_grid.pdf (and .png).
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
from fabricnet.explain import GradCAM, overlay, spectral_maps  # noqa: E402
from fabricnet.models import SpecTexNet, build_model  # noqa: E402


def load_run(run_dir):
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    cfg, classes = ck["cfg"], ck["classes"]
    m = cfg["model"]
    model = build_model(m["name"], len(classes), pretrained=False,
                        **{k: v for k, v in m.items() if k not in {"name", "pretrained"}})
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="spectex_convnext_tiny")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures/explain_grid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.results) / a.exp / f"r0_f{a.fold}"
    model, cfg, classes = load_run(run)
    preds = pd.read_csv(run / "test_predictions.csv")
    gray = cfg["data"].get("grayscale", False)
    rng = np.random.default_rng(a.seed)
    rows = []
    for yi, c in enumerate(classes):
        if c == "defect_free":
            continue
        sub = preds[(preds.y == yi) & (preds.pred == yi)]  # prefer correctly classified examples
        if not len(sub):
            sub = preds[preds.y == yi]
        if len(sub):
            rows += sub.sample(min(a.per_class, len(sub)), random_state=int(rng.integers(1e6))).to_dict("records")
    tf = build_transforms(cfg["data"]["img_size"], train=False)
    cam = GradCAM(model)
    is_spec = isinstance(model, SpecTexNet) and model.fusion != "none"
    cols = ["Input", "Grad-CAM"] + (["Spectral saliency", "SGA gate"] if is_spec else [])
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.0 * len(cols), 2.0 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        img = Image.open(r["path"])
        img = (img.convert("L") if gray else img).convert("RGB").resize((cfg["data"]["img_size"],) * 2)
        x = tf(img)[None]
        rgb = np.asarray(img)
        heat, _ = cam(x.clone(), target=int(r["y"]))
        panels = [rgb, overlay(rgb, heat)]
        if is_spec:
            sm = spectral_maps(model, x)
            key = [k for k in sm if k.startswith("spectral_")][-1]
            panels += [overlay(rgb, sm[key]), overlay(rgb, sm["sga_gate"])]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(p)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
        axes[i, 0].set_ylabel(classes[int(r["y"])].replace("_", " "), fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=300)
    fig.savefig(a.out + ".png", dpi=150)
    print(f"saved {a.out}.pdf")


if __name__ == "__main__":
    main()
