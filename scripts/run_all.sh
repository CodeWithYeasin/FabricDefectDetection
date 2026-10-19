#!/usr/bin/env bash
# Full experimental pipeline. Expect roughly 10-14 GPU-hours on one T4 / ~4-5 h on an A100
# for 17 experiments x 5 folds at 224 px. Each step is resumable (--skip-done).
#
#   bash scripts/run_all.sh "data/raw/Fabric Defects Dataset"
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data/raw/Fabric Defects Dataset}"

# 1) index + de-duplicate + leakage-free folds
[ -f data/splits.csv ] || python scripts/prepare_data.py --roots "$DATA_ROOT" --out data

# 2) shortcut audit, v1 pipeline on the clean folds and on a deliberately leaky split (CPU is fine)
python scripts/shortcut_check.py
python scripts/run_legacy_v1.py
[ -f data_leaky/splits.csv ] || python scripts/prepare_data.py --roots "$DATA_ROOT" --out data_leaky \
    --derived keep --policy keep_all --no-groups --max-aspect 1000
python scripts/run_legacy_v1.py --prefix leaky_ --set data.splits_csv=data_leaky/splits.csv data.grayscale=false

# 3) proposed model first, then baselines, then ablations
EXPS=(spectex_convnext_tiny spectex_resnet50 spectex_efficientnet_b0
      resnet50 efficientnet_b0 mobilenetv3_large_100 convnext_tiny swin_tiny deit_small
      abl_no_spectral abl_concat_fusion abl_fixed_sr abl_single_scale
      abl_aug_flip abl_aug_naive_rot abl_loss_ce abl_no_derived abl_colour)
for e in "${EXPS[@]}"; do
  python scripts/train.py --exp "$e" --skip-done
done

# 4) efficiency, figures, statistics, LaTeX tables
python scripts/benchmark_efficiency.py
python scripts/figure_method.py
python scripts/explain.py --exp spectex_convnext_tiny --fold 0 --per-class 2
python scripts/aggregate.py --proposed spectex_convnext_tiny

# 5) deployment model trained on all data
python scripts/train.py --exp spectex_convnext_tiny --full
python scripts/export_model.py --run results/spectex_convnext_tiny/full
echo "done. Compile paper/main.tex; push deploy/hf_space to your Space."
