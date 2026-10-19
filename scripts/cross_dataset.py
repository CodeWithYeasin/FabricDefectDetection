#!/usr/bin/env python
"""Zero-shot cross-dataset generalisation: models trained on the source dataset (all CV folds,
averaged as an ensemble) are tested on an external dataset WITHOUT any fine-tuning.

Class names rarely line up across fabric datasets, so we report:
  1. binary defect vs. defect-free metrics (always meaningful), and
  2. multi-class metrics over the classes the mapping file can align.

python scripts/cross_dataset.py --exp spectex_convnext_tiny --root data/external/TILDA \
       --mapping configs/mappings/tilda.yaml --name tilda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fabricnet.data import FabricDataset, build_index  # noqa: E402
from fabricnet.engine import predict  # noqa: E402
from fabricnet.models import build_model  # noqa: E402
from fabricnet.utils import get_logger, save_json  # noqa: E402

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mapping", required=True, help="yaml: aliases {folder: class}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    with open(a.mapping) as f:
        mp = yaml.safe_load(f)
    runs = sorted((Path(a.results) / a.exp).glob("r0_f*/best.pt"))
    if not runs:
        raise SystemExit("no trained folds found")
    ck0 = torch.load(runs[0], map_location="cpu", weights_only=False)
    classes, cfg = ck0["classes"], ck0["cfg"]
    df, _ = build_index([a.root], aliases=mp["aliases"], classes=classes)
    log.info(f"external set: {len(df)} images\n{df.label.value_counts()}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(FabricDataset(df, cfg["data"]["img_size"], grayscale=cfg["data"].get("grayscale", False)), batch_size=32, num_workers=cfg["data"]["num_workers"])
    probs = []
    for r in runs:
        ck = torch.load(r, map_location="cpu", weights_only=False)
        m = ck["cfg"]["model"]
        model = build_model(m["name"], len(classes), pretrained=False,
                            **{k: v for k, v in m.items() if k not in {"name", "pretrained"}})
        model.load_state_dict(ck["model"])
        probs.append(predict(model.to(dev), dl, dev, tta=cfg.get("eval", {}).get("tta", "none"), classes=classes))
    P = np.mean(probs, 0)
    y = df["y"].values
    n = classes.index("defect_free")
    yb, pb = (y != n).astype(int), 1 - P[:, n]
    out = {"n": len(df), "class_counts": df.label.value_counts().to_dict(), "n_models": len(runs),
           "binary": {"auroc": float(roc_auc_score(yb, pb)) if 0 < yb.sum() < len(yb) else None,
                      "balanced_accuracy": float(balanced_accuracy_score(yb, (pb > 0.5).astype(int))),
                      "f1_defect": float(f1_score(yb, (pb > 0.5).astype(int)))}}
    present = sorted(set(y))
    Pm = P[:, present]
    pred = np.array(present)[Pm.argmax(1)]
    out["multiclass_aligned"] = {"classes": [classes[i] for i in present],
                                 "macro_f1": float(f1_score(y, pred, average="macro")),
                                 "balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    save_json(out, Path(a.results) / f"cross_dataset_{a.name}_{a.exp}.json")
    log.info(out)


if __name__ == "__main__":
    main()
