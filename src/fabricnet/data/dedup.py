"""Exact and near-duplicate detection with perceptual hashes.

Public fabric-defect datasets on Kaggle often contain re-saved, resized or lightly edited
copies of the same photo. If one copy lands in train and another in test, the reported
accuracy is inflated. We hash every image (pHash + dHash), link images whose hashes are
within a Hamming threshold, and use the connected components as *groups* for splitting,
so near-duplicates always stay in the same fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    a = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    med = np.median(d[1:, 1:]) if hash_size > 1 else np.median(d)
    return (d > med).flatten()


def dhash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.float64)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8), axis=-1)


def compute_hashes(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ph, dh = [], []
    for p in paths:
        with Image.open(p) as im:
            ph.append(phash(im))
            dh.append(dhash(im))
    return _pack(np.stack(ph)), _pack(np.stack(dh))


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(packed: np.ndarray, block: int = 1024) -> np.ndarray:
    n = packed.shape[0]
    out = np.empty((n, n), dtype=np.uint8)
    for i in range(0, n, block):
        x = packed[i : i + block, None, :] ^ packed[None, :, :]
        out[i : i + block] = _POP[x].sum(-1)
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def find_duplicate_groups(
    df: pd.DataFrame, phash_thr: int = 4, dhash_thr: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds ``group`` column. Two images are linked if md5 matches, or BOTH hash distances
    are under threshold (requiring both keeps false merges of similar plain textures low).

    Returns ``(df_with_group, pairs)`` where ``pairs`` lists every linked pair for the audit log.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    ph, dh = compute_hashes(df["path"].tolist())
    dp, dd = hamming_matrix(ph), hamming_matrix(dh)
    uf = _UF(n)
    near = (dp <= phash_thr) & (dd <= dhash_thr)
    np.fill_diagonal(near, False)
    ii, jj = np.where(np.triu(near))
    pairs = []
    for i, j in zip(ii, jj):
        uf.union(i, j)
        pairs.append((i, j, int(dp[i, j]), int(dd[i, j]), "near"))
    for _, idx in df.groupby("md5").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
            pairs.append((idx[0], k, 0, 0, "exact"))
    df["group"] = [uf.find(i) for i in range(n)]
    pairs_df = pd.DataFrame(pairs, columns=["i", "j", "phash_dist", "dhash_dist", "kind"])
    if len(pairs_df):
        pairs_df["path_i"] = df.loc[pairs_df.i, "path"].values
        pairs_df["path_j"] = df.loc[pairs_df.j, "path"].values
        pairs_df["label_i"] = df.loc[pairs_df.i, "label"].values
        pairs_df["label_j"] = df.loc[pairs_df.j, "label"].values
    return df, pairs_df


def resolve_groups(df: pd.DataFrame, policy: str = "keep_one") -> tuple[pd.DataFrame, dict]:
    """Handle duplicate groups.

    - groups whose members disagree on the label are dropped entirely (label noise);
    - ``keep_one``: keep the first member of each remaining group (cleanest; default);
    - ``keep_all``: keep everything, rely on group-aware splitting.
    """
    stats = {"n_before": len(df)}
    n_labels = df.groupby("group")["label"].nunique()
    conflict = n_labels[n_labels > 1].index
    stats["conflicting_groups"] = int(len(conflict))
    stats["dropped_conflicting_images"] = int(df["group"].isin(conflict).sum())
    df = df[~df["group"].isin(conflict)]
    sizes = df.groupby("group").size()
    stats["duplicate_groups"] = int((sizes > 1).sum())
    stats["images_in_duplicate_groups"] = int(sizes[sizes > 1].sum())
    if policy == "keep_one":
        df = df.drop_duplicates("group", keep="first")
    stats["n_after"] = len(df)
    return df.reset_index(drop=True), stats
