#!/usr/bin/env python
"""How far can you get WITHOUT looking at defects? (source-shortcut audit)

In the Kaggle Fabric Defects Dataset the classes come from different sources: most defect-free
images are 4096x256 grey line-scan strips, most defect images are 640x360 colour photos, some
are phone photos, some are binarised copies. A classifier can exploit that instead of learning
defects. This script trains a random forest on the same folds using only
  (a) file metadata: width, height, aspect, colour mode, file size, number of grey levels
  (b) global statistics of the image the CNN actually sees after our protocol
      (grey, centre-cropped if elongated, resized): mean, std, percentiles, sharpness
High macro-F1 for (a) shows that a shortcut exists in the raw data. A low score for (b) shows
that the protocol removed most of it. Report both in the paper.

python scripts/shortcut_check.py   ->  results/shortcut_check.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

from fabricnet.data import resolve_paths  # noqa: E402
from fabricnet.utils import get_logger, save_json  # noqa: E402

log = get_logger()


def meta_features(df):
    mode_rgb = (df["mode"] == "RGB").astype(float)
    return np.c_[df["width"], df["height"], df["width"] / df["height"].clip(lower=1), mode_rgb,
                 np.log1p(df["filesize"]), df["grey_levels"]]


def pixel_features(paths, size=224):
    out = []
    for p in paths:
        g = Image.open(p).convert("L").resize((size, size))
        a = np.asarray(g, dtype=np.float32)
        lap = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        out.append([a.mean(), a.std(), *np.percentile(a, [5, 50, 95]), lap.mean(), lap.std()])
    return np.array(out)


def cv_score(X, df, seed=0):
    preds = np.zeros(len(df), dtype=int)
    for k in sorted(df["fold"].unique()):
        tr, te = df["fold"].values != k, df["fold"].values == k
        rf = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(X[tr], df["y"].values[tr])
        preds[te] = rf.predict(X[te])
    y = df["y"].values
    return {"macro_f1": float(f1_score(y, preds, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "accuracy": float((preds == y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--out", default="results/shortcut_check.json")
    a = ap.parse_args()
    sp = Path(a.splits)
    df = resolve_paths(pd.read_csv(sp if sp.is_absolute() else ROOT / sp), ROOT)
    df = df[df["kind"].ne("derived")].reset_index(drop=True) if "kind" in df else df
    res = {"n": len(df)}
    res["metadata_only"] = cv_score(meta_features(df), df)
    log.info(f"metadata-only random forest: {res['metadata_only']}")
    res["processed_input_stats"] = cv_score(pixel_features(df["path"]), df)
    log.info(f"global statistics of the protocol input: {res['processed_input_stats']}")
    save_json(res, ROOT / a.out if not Path(a.out).is_absolute() else a.out)


if __name__ == "__main__":
    main()
