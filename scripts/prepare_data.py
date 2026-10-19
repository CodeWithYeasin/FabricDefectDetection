#!/usr/bin/env python
"""Index -> provenance (originals / derived copies / non-photos) -> de-duplicate -> folds -> cache.

Example
-------
python scripts/prepare_data.py --roots "/kaggle/input/fabric-defects-dataset" --out data

Outputs (in --out)
  splits.csv             one row per usable image. Columns include path (cached, relative to repo),
                         orig_path, label, y, kind (original|derived), group, fold
  dedup_report.json      counts: raw, non-photos dropped, derived copies, duplicates, per class/fold
  duplicate_pairs.csv    every hash-linked pair. Inspect some by eye before trusting the thresholds
  name_label_conflicts.csv  files whose name says one class but whose folder says another
  image_stats_by_class.csv  size / colour mode per class (to spot source shortcuts)
  skipped.csv            files that were not images of a known class
  cache/                 resized copies used for training (short side --cache-size)

Derived policy (--derived)
  train_only  (default) derived copies stay in the fold of their source, are used for training,
              and are never used for validation or testing
  drop        remove them completely
  keep        treat them like originals (reproduces the v1 leakage; only for the leakage study)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from fabricnet import CLASSES  # noqa: E402
from fabricnet.data import build_index, check_no_leakage, find_duplicate_groups, make_folds, resolve_groups  # noqa: E402
from fabricnet.data.cache import cache_image  # noqa: E402
from fabricnet.utils import get_logger, save_json  # noqa: E402

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phash-thr", type=int, default=4)
    ap.add_argument("--dhash-thr", type=int, default=6)
    ap.add_argument("--policy", choices=["keep_one", "keep_all"], default="keep_one")
    ap.add_argument("--derived", choices=["train_only", "drop", "keep"], default="train_only")
    ap.add_argument("--keep-nonphoto", action="store_true")
    ap.add_argument("--no-groups", action="store_true",
                    help="ignore duplicate groups when splitting (ONLY for the leakage study that mimics v1)")
    ap.add_argument("--repeats", type=int, default=1, help="extra fold assignments (fold_r1, ...)")
    ap.add_argument("--cache-size", type=int, default=320, help="short side of cached images; 0 disables cache")
    ap.add_argument("--max-aspect", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, skipped = build_index(args.roots)
    if df.empty:
        raise SystemExit("No images found. Check --roots and folder names (see DEFAULT_ALIASES).")
    skipped.to_csv(out / "skipped.csv", index=False)
    stats = {"n_raw": len(df), "raw_class_counts": df["label"].value_counts().to_dict(),
             "raw_kind_counts": df.groupby(["label", "kind"]).size().unstack(fill_value=0).to_dict()}
    log.info(f"indexed {len(df)} images, skipped {len(skipped)}\n"
             + df.groupby(["label", "kind"]).size().unstack(fill_value=0).to_string())

    df.groupby("label").agg(n=("path", "size"), median_w=("width", "median"), median_h=("height", "median"),
                            frac_grey=("mode", lambda m: float((m == "L").mean())),
                            median_kb=("filesize", lambda s: float(s.median() / 1024))).to_csv(
        out / "image_stats_by_class.csv")
    conflicts = df[(df["name_label"] != "") & (df["name_label"] != df["label"])]
    conflicts[["path", "label", "name_label"]].to_csv(out / "name_label_conflicts.csv", index=False)
    stats["name_label_conflicts"] = int(len(conflicts))
    if len(conflicts):
        log.warning(f"{len(conflicts)} files have a filename that suggests another class; "
                    f"see {out / 'name_label_conflicts.csv'} (kept with their folder label)")

    # ---- non-photographs (and their copies) ----
    nonphoto_keys = set(df.loc[df["kind"] == "nonphoto", "base_key"])
    drop_np = (df["kind"] == "nonphoto") | (df["derived"] & df["base_key"].isin(nonphoto_keys))
    stats["nonphoto_dropped"] = 0 if args.keep_nonphoto else int(drop_np.sum())
    if not args.keep_nonphoto:
        df = df[~drop_np]
    df.loc[df["kind"] == "nonphoto", "kind"] = "original"

    # ---- derived copies ----
    if args.derived == "keep":
        df["kind"] = "original"
    orig = df[df["kind"] == "original"].reset_index(drop=True)
    der = df[df["kind"] == "derived"].reset_index(drop=True)
    stats["derived_copies"] = int(len(der))

    # ---- hash de-duplication among originals ----
    orig, pairs = find_duplicate_groups(orig, args.phash_thr, args.dhash_thr)
    pairs.to_csv(out / "duplicate_pairs.csv", index=False)
    key2group = dict(zip(orig["base_key"], orig["group"]))
    orig, dstats = resolve_groups(orig, args.policy)
    stats.update(dstats)
    stats["exact_pairs"] = int((pairs["kind"] == "exact").sum()) if len(pairs) else 0
    stats["near_pairs"] = int((pairs["kind"] == "near").sum()) if len(pairs) else 0
    stats["cross_label_pairs"] = int((pairs["label_i"] != pairs["label_j"]).sum()) if len(pairs) else 0

    if args.no_groups:
        orig["group"] = range(len(orig))
    orig = make_folds(orig, args.n_splits, args.seed)
    for r in range(1, args.repeats):
        orig[f"fold_r{r}"] = make_folds(orig, args.n_splits, args.seed + r)["fold"].values
    fold_cols = [c for c in orig.columns if c == "fold" or c.startswith("fold_r")]

    if args.derived == "drop" or der.empty:
        der = der.iloc[0:0]
    else:
        der["group"] = der["base_key"].map(key2group)
        der = der[der["group"].isin(set(orig["group"]))]  # source survived de-dup / conflicts
        gmap = orig.drop_duplicates("group").set_index("group")[fold_cols]
        der = der.join(gmap, on="group")
    df = pd.concat([orig, der], ignore_index=True)
    check_no_leakage(df)
    stats["clean_class_counts"] = orig["label"].value_counts().to_dict()
    stats["train_only_derived_counts"] = der["label"].value_counts().to_dict()
    stats["n_after"] = int(len(orig))
    stats["fold_class_counts"] = orig.groupby(["fold", "label"]).size().unstack(fill_value=0).to_dict()

    # ---- resized cache (paths relative to the repo root) ----
    df["orig_path"] = df["path"]
    if args.cache_size > 0:
        cdir = out / "cache"
        new_paths, cropped = [], 0
        for p, md5 in tqdm(zip(df["orig_path"], df["md5"]), total=len(df), desc="caching"):
            dst = cdir / f"{md5}.jpg"
            if not dst.exists():
                _, _, c = cache_image(p, dst, args.cache_size, args.max_aspect)
                cropped += int(c)
            try:
                new_paths.append(str(dst.resolve().relative_to(ROOT)))
            except ValueError:
                new_paths.append(str(dst.resolve()))
        df["path"] = new_paths
        stats["cache_short_side"] = args.cache_size
        stats["center_cropped_elongated"] = cropped
    stats.update({"phash_thr": args.phash_thr, "dhash_thr": args.dhash_thr, "policy": args.policy,
                  "derived_policy": args.derived, "classes": CLASSES})
    df.to_csv(out / "splits.csv", index=False)
    save_json(stats, out / "dedup_report.json")
    log.info("clean originals per class: " + str(stats["clean_class_counts"]))
    log.info(f"wrote {out / 'splits.csv'}")


if __name__ == "__main__":
    main()
