#!/usr/bin/env python
"""Draw the SpecTexNet block diagram -> paper/figures/architecture.pdf (vector, editable)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper/figures/architecture"

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="0.25", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1, color="0.2"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, text, ha="center", fontsize=7, color="0.3")


C_IN, C_BB, C_SP, C_F, C_H = "#eeeeee", "#cfe2f3", "#fde9c8", "#d9ead3", "#ead1dc"
box(1, 15, 9, 6, "RGB image\n224×224", C_IN)
box(16, 25, 22, 7, "CNN backbone\n(ConvNeXt-T / ResNet50 / EffNet-B0)", C_BB)
box(16, 3, 22, 9, "Learnable spectral residual\nlog|FFT| − softmax(θ)∗log|FFT|\nscales 64, 128", C_SP)
box(43, 3, 14, 9, "Spectral encoder\n4× conv-BN-SiLU, s2", C_SP)
box(62, 13, 15, 11, "SGA\n$a=σ(W[φ(F);G])$\n$m=σ(\\mathrm{MLP}(\\bar G))$\n$F'=F⊙(1+a)⊙(1+m)$", C_F)
box(81, 13, 8, 11, "GAP(F')\n⊕\nGAP(G)", C_F)
box(92, 15.5, 7, 6, "LN, FC\n6 classes", C_H)
arrow(10, 19, 16, 28)
arrow(10, 17, 16, 7.5, "grey")
arrow(38, 28.5, 62, 21, "F (C×7×7)")
arrow(38, 7.5, 43, 7.5)
arrow(57, 7.5, 62, 16, "G (128×7×7)")
arrow(77, 18.5, 81, 18.5)
arrow(89, 18.5, 92, 18.5)
ax.text(27, 0.2, "OA-Rot (train): rot k·90°, swap horizontal↔vertical when k is odd", fontsize=7, ha="center", color="0.35")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT) + ".pdf")
fig.savefig(str(OUT) + ".png", dpi=200)
print("saved", OUT)
