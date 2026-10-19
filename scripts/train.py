#!/usr/bin/env python
"""Run one experiment over all CV folds (and repeats).

Examples
--------
python scripts/train.py --exp spectex_convnext_tiny
python scripts/train.py --exp resnet50 --folds 0 1
python scripts/train.py --exp abl_aug_flip --set train.epochs=30
python scripts/train.py --exp spectex_convnext_tiny --full      # train on ALL data for deployment

Outputs: results/<exp>/r<rep>_f<fold>/{best.pt,history.csv,metrics.json,test_predictions.csv,...}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fabricnet.data import fold_frames, inner_split, resolve_paths  # noqa: E402
from fabricnet.engine import train_fold  # noqa: E402
from fabricnet.utils import (  # noqa: E402
    apply_overrides as load_config_overrides,
    get_logger,
    load_config,
    load_json,
    save_json,
    seed_everything,
)

log = get_logger()


def experiment_config(exp: str, extra: list[str] | None = None, base: str = "configs/default.yaml") -> dict:
    with open(ROOT / "configs/experiments.yaml") as f:
        exps = yaml.safe_load(f)["experiments"]
    if exp not in exps:
        raise SystemExit(f"unknown experiment {exp!r}; choose from {list(exps)}")
    cfg = load_config(ROOT / base)
    for key, val in exps[exp].items():
        if key == "group":
            continue
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    cfg = load_config_overrides(cfg, extra or [])
    cfg["experiment"] = exp
    cfg["groups"] = exps[exp].get("group", [])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    ap.add_argument("--full", action="store_true", help="train on all data (epochs = median best epoch of CV)")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--deadline", type=float, default=0,
                    help="unix time; no new fold is started after it (Kaggle 12 h session limit)")
    args = ap.parse_args()

    cfg = experiment_config(args.exp, args.set)
    classes = cfg["classes"]
    sp = Path(cfg["data"]["splits_csv"])
    df = resolve_paths(pd.read_csv(sp if sp.is_absolute() else ROOT / sp), ROOT)
    use_der = cfg["data"].get("use_derived", True)
    out_root = Path(cfg["output_root"]) / args.exp
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.full:
        mets = sorted(out_root.glob("r*_f*/metrics.json"))
        if mets:
            eps = [load_json(m)["best_epoch"] for m in mets]
            cfg["train"]["epochs"] = int(np.median(eps)) + 1
            log.info(f"--full: using {cfg['train']['epochs']} epochs (median best epoch of {len(eps)} CV runs)")
        cfg["train"]["patience"] = 10**9
        seed_everything(cfg["seed"])
        is_der = df["kind"].eq("derived") if "kind" in df else pd.Series(False, index=df.index)
        _, va = inner_split(df[~is_der].reset_index(drop=True), cfg["data"]["val_frac"], cfg["seed"])
        train_all = df if use_der else df[~is_der]
        train_fold(cfg, train_all.reset_index(drop=True), va, None, out_root / "full", classes)  # va only monitors
        return

    repeats = args.repeats or cfg["cv"].get("repeats", 1)
    n_splits = cfg["cv"]["n_splits"]
    folds = args.folds if args.folds is not None else list(range(n_splits))
    summary = {}
    for r in range(repeats):
        col = "fold" if r == 0 else f"fold_r{r}"
        if col not in df:
            raise SystemExit(f"{col} missing in splits.csv; rerun prepare_data.py --repeats {repeats}")
        for k in folds:
            run = out_root / f"r{r}_f{k}"
            if args.skip_done and (run / "metrics.json").exists():
                summary[run.name] = load_json(run / "metrics.json")
                continue
            if args.deadline and time.time() > args.deadline:
                log.warning(f"deadline reached, stopping before {run.name}; rerun with --skip-done to continue")
                save_json(summary, out_root / "cv_runs.json")
                return
            seed_everything(cfg["seed"] + 100 * r + k)
            tr, va, test = fold_frames(df, k, col, cfg["data"]["val_frac"], cfg["seed"] + k, use_der)
            log.info(f"[{args.exp}] rep {r} fold {k}: train {len(tr)} val {len(va)} test {len(test)}")
            res = train_fold(cfg, tr, va, test, run, classes)
            summary[run.name] = res
            keep = cfg["train"].get("keep_ckpt", "fold0")
            if keep == "none" or (keep == "fold0" and not (r == 0 and k == 0)):
                (run / "best.pt").unlink(missing_ok=True)  # saves disk on Kaggle (20 GB limit)
            log.info(f"[{args.exp}] {run.name} test macro-F1 {res['test']['macro_f1']:.4f} acc {res['test']['accuracy']:.4f}")
    save_json(summary, out_root / "cv_runs.json")


if __name__ == "__main__":
    main()
