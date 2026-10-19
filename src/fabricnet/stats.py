"""Uncertainty and significance testing.

- ``bootstrap_ci``      stratified percentile bootstrap CI for any metric on pooled OOF predictions
- ``mcnemar``           paired test between two classifiers on the same images (exact binomial)
- ``friedman_nemenyi``  multi-model comparison over CV folds, with Nemenyi critical difference
- ``wilcoxon_holm``     pairwise per-fold Wilcoxon signed-rank vs. a reference, Holm-corrected

Note: with 5 folds the Wilcoxon test cannot reach p<0.05 (minimum two-sided p = 0.0625).
Use 5x2 repeated CV (``repeats`` in configs) or rely on McNemar / bootstrap over pooled
out-of-fold predictions, which use every image.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as ss

# Nemenyi q_alpha (alpha=0.05) for k=2..12 models (Demšar, JMLR 2006)
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        11: 3.219, 12: 3.268}


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, metric: Callable, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_by_c = [np.where(y == c)[0] for c in np.unique(y)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_c])
        vals.append(metric(y[idx], probs[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, probs)), float(lo), float(hi)


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * ss.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


def friedman_nemenyi(scores: np.ndarray, names: list[str]) -> dict:
    """scores: (n_folds, k_models), higher is better."""
    n, k = scores.shape
    ranks = np.vstack([ss.rankdata(-row) for row in scores])  # rank 1 = best
    avg = ranks.mean(0)
    if k >= 3 and n >= 2:
        stat, p = ss.friedmanchisquare(*scores.T)
    else:
        stat, p = float("nan"), float("nan")
    cd = _Q05.get(k, float("nan")) * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"avg_rank": dict(zip(names, avg.tolist())), "friedman_chi2": float(stat), "p_value": float(p),
            "nemenyi_cd": float(cd), "n_folds": n}


def wilcoxon_holm(scores: np.ndarray, names: list[str], ref: str) -> dict:
    j = names.index(ref)
    raw = {}
    for i, nm in enumerate(names):
        if i == j:
            continue
        d = scores[:, j] - scores[:, i]
        if np.allclose(d, 0):
            raw[nm] = 1.0
        else:
            raw[nm] = float(ss.wilcoxon(scores[:, j], scores[:, i], zero_method="zsplit").pvalue)
    order = sorted(raw, key=raw.get)
    m, adj, running = len(order), {}, 0.0
    for r, nm in enumerate(order):
        running = max(running, min(1.0, (m - r) * raw[nm]))
        adj[nm] = running
    return {"reference": ref, "p_raw": raw, "p_holm": adj}


def plot_cd_diagram(avg_rank: dict, cd: float, path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(avg_rank, key=avg_rank.get)
    r = [avg_rank[n] for n in names]
    k = len(names)
    fig, ax = plt.subplots(figsize=(7, 0.45 * k + 1.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5, k)
    ax.invert_xaxis()
    ax.axhline(k - 0.3, color="k", lw=0.8)
    for t in range(1, k + 1):
        ax.plot([t, t], [k - 0.4, k - 0.2], color="k", lw=0.8)
        ax.text(t, k - 0.1, str(t), ha="center", va="bottom", fontsize=8)
    for i, (nm, rk) in enumerate(zip(names, r)):
        y = k - 1.2 - i
        ax.plot([rk, rk], [k - 0.3, y], color="0.3", lw=0.8)
        ax.text(rk, y - 0.05, f"{nm} ({rk:.2f})", ha="center", va="top", fontsize=8)
    ax.plot([r[0], r[0] + cd], [k + 0.3, k + 0.3], color="C3", lw=2)
    ax.text(r[0] + cd / 2, k + 0.35, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=8, color="C3")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
