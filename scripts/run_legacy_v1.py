#!/usr/bin/env python
"""Re-evaluate the v1 pipeline under the new, leakage-free protocol (same folds as everything else).

Produces three experiments in the same format as scripts/train.py:
  v1_frozen_r50_gbdt_ens   frozen ResNet50 + scratch GBDT A/B/C + val-fitted weights (the v1 system, bug fixed)
  v1_frozen_r50_gbdt_a     single scratch GBDT
  v1_frozen_r50_linear     frozen ResNet50 + logistic-regression linear probe (standard reference)

Features are cached in results/_cache/r50_feats.npz keyed by image path.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fabricnet.data import FabricDataset, fold_frames, resolve_paths  # noqa: E402
from fabricnet.legacy import V1_CONFIGS, ScratchGBDT, fit_weights  # noqa: E402
from fabricnet.metrics import compute_metrics  # noqa: E402
from fabricnet.utils import get_logger, load_config, save_json  # noqa: E402

log = get_logger()


@torch.no_grad()
def extract(df, img_size, pretrained=True, bs=64, workers=4, grayscale=True):
    import timm

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, num_classes=0).to(dev).eval()
    dl = DataLoader(FabricDataset(df, img_size, train=False, grayscale=grayscale), batch_size=bs, num_workers=workers)
    out = []
    for x, _, _ in dl:
        out.append(m(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def save_run(run_dir, test, probs, classes, extra=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf = test[["path", "y"]].copy()
    pdf["pred"] = probs.argmax(1)
    for i, c in enumerate(classes):
        pdf[f"p_{c}"] = probs[:, i]
    pdf.to_csv(run_dir / "test_predictions.csv", index=False)
    res = {"test": compute_metrics(test["y"].values, probs, classes), "best_epoch": -1}
    res.update(extra or {})
    save_json(res, run_dir / "metrics.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-estimators", type=int, default=70)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--prefix", default="", help="e.g. 'leaky_' for the leakage study on a leaky split file")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config, args.set)
    classes = cfg["classes"]
    sp = Path(cfg["data"]["splits_csv"])
    df = resolve_paths(pd.read_csv(sp if sp.is_absolute() else ROOT / sp), ROOT)
    out_root = Path(cfg["output_root"])
    gs = "gray" if cfg["data"].get("grayscale", True) else "rgb"
    cache = out_root / "_cache" / f"r50_feats_{gs}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feat_map = dict(zip(z["paths"], z["X"]))
    else:
        feat_map = {}
    missing = df[~df["path"].isin(feat_map)]
    if len(missing):
        log.info(f"extracting ResNet50 features for {len(missing)} images")
        X = extract(missing, cfg["data"]["img_size"], not args.no_pretrained, workers=cfg["data"]["num_workers"],
                    grayscale=cfg["data"].get("grayscale", True))
        feat_map.update(zip(missing["path"], X))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, paths=np.array(list(feat_map)), X=np.stack(list(feat_map.values())))

    def feats(d):
        return np.stack([feat_map[p] for p in d["path"]])

    for k in range(cfg["cv"]["n_splits"]):
        tr, va, test = fold_frames(df, k, "fold", cfg["data"]["val_frac"], cfg["seed"] + k,
                                   cfg["data"].get("use_derived", True))
        Xtr, Xva, Xte = feats(tr), feats(va), feats(test)
        ytr, yva = tr["y"].values, va["y"].values
        run = f"r0_f{k}"

        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        r = save_run(out_root / (args.prefix + "v1_frozen_r50_linear") / run, test, lr.predict_proba(sc.transform(Xte)), classes)
        log.info(f"fold {k} linear probe mF1 {r['test']['macro_f1']:.4f}")

        models = {n: ScratchGBDT(args.n_estimators, 0.1, args.max_depth, seed=cfg["seed"] + k, **c).fit(Xtr, ytr)
                  for n, c in V1_CONFIGS.items()}
        pva = [m.predict_proba(Xva) for m in models.values()]
        pte = [m.predict_proba(Xte) for m in models.values()]
        w, s = fit_weights(pva, yva, seed=cfg["seed"])
        r = save_run(out_root / (args.prefix + "v1_frozen_r50_gbdt_a") / run, test, pte[0], classes)
        log.info(f"fold {k} gbdt_a mF1 {r['test']['macro_f1']:.4f}")
        ens = sum(wi * p for wi, p in zip(w, pte))
        r = save_run(out_root / (args.prefix + "v1_frozen_r50_gbdt_ens") / run, test, ens, classes,
                     {"weights": w.tolist(), "val_macro_f1": s})
        log.info(f"fold {k} gbdt ensemble mF1 {r['test']['macro_f1']:.4f} weights {np.round(w, 3)}")


if __name__ == "__main__":
    main()
