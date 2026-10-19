"""Leakage-free, stratified, group-aware cross-validation splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign ``fold`` in [0, n_splits). Groups never cross folds; class ratios are preserved."""
    df = df.copy()
    groups = df["group"] if "group" in df else np.arange(len(df))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["y"], groups)):
        df.iloc[te, df.columns.get_loc("fold")] = k
    assert (df["fold"] >= 0).all()
    return df


def inner_split(train_df: pd.DataFrame, val_frac: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve an early-stopping/validation subset out of the training folds (group-aware)."""
    n_splits = max(2, int(round(1 / val_frac)))
    groups = train_df["group"] if "group" in train_df else np.arange(len(train_df))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(train_df, train_df["y"], groups))
    return train_df.iloc[tr].reset_index(drop=True), train_df.iloc[va].reset_index(drop=True)


def fold_frames(df: pd.DataFrame, k: int, col: str = "fold", val_frac: float = 0.1, seed: int = 0,
                use_derived: bool = True):
    """Train / val / test frames for test fold ``k``.

    Validation and test contain ORIGINAL images only. Derived copies (``kind == 'derived'``) of
    training-fold images are appended to the training frame when ``use_derived`` is set.
    """
    is_der = df["kind"].eq("derived") if "kind" in df else pd.Series(False, index=df.index)
    orig, der = df[~is_der], df[is_der]
    test = orig[orig[col] == k].reset_index(drop=True)
    tr, va = inner_split(orig[orig[col] != k].reset_index(drop=True), val_frac, seed)
    if use_derived and len(der):
        val_groups = set(va["group"]) if "group" in va else set()
        extra = der[(der[col] != k) & ~der["group"].isin(val_groups)]
        tr = pd.concat([tr, extra], ignore_index=True)
    return tr, va, test


def resolve_paths(df: pd.DataFrame, root) -> pd.DataFrame:
    """Make relative ``path`` entries absolute with respect to the repository root."""
    from pathlib import Path

    df = df.copy()
    df["path"] = [p if Path(p).is_absolute() else str(Path(root) / p) for p in df["path"]]
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    if "group" not in df:
        return
    per_group = df.groupby("group")["fold"].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} duplicate groups span multiple folds")
