# Audit of the v1 pipeline

This file records what was wrong or weak in v1 (notebooks in `legacy/notebooks/`) and what v2
does about it. Reviewers often ask how earlier numbers relate to new ones, and this is the answer.

## What v1 did

1. Downloaded the Kaggle Fabric Defects Dataset (6 classes, 2739 images used) and a second small
   dataset (`fabricfaultdetection`, 79 images).
2. Extracted 2048-d features with a frozen ImageNet ResNet50 (`torchvision IMAGENET1K_V1`).
3. Split 80/20 at random into train and test, then 80/20 again for validation (1756 / 435 / 548).
4. Trained three NumPy gradient-boosting models written from scratch and combined them with
   weights found by random search.
5. Reported 93.25% test accuracy (notebook 08) and deployed the notebook 10 models to a
   Streamlit app on Hugging Face Spaces.

## Problems found

| # | Problem | Where | Effect | v2 fix |
|---|---------|-------|--------|--------|
| 1 | Ensemble weights fitted on the training data (train+val), "Best Ensemble Train Acc: 1.0000" | nb 10, cell 7 | Deployed weights carry no information about generalisation | `legacy/gbdt.py::fit_weights` is fitted on the inner validation split only |
| 2 | The deployed model (nb 10) was never evaluated on the test set | nb 10/11 | Reported 93.25% belongs to a different model (nb 08) | Every model is evaluated on the same held-out folds |
| 3 | Single random split, no repeated runs | nb 05 | No variance, no confidence interval, no significance test possible | 5-fold stratified group CV, bootstrap CI, McNemar, Friedman/Nemenyi |
| 4 | Accuracy as the headline metric with ~61% defect-free images | nb 07-09 | A model that always says "defect-free" gets ~61% | Macro-F1 (headline), balanced accuracy, MCC, AUROC, ECE, detection/false-alarm rate |
| 5 | Minority classes were weak but hidden by accuracy: F1 0.65 (class 0, Vertical), 0.73 (hole), 0.71-0.84 (horizontal) | `legacy/metrics/evaluation_results.json` | Weakest exactly on the defects that matter | Class-balanced focal loss, OA-Rot, spectral branch, per-class reporting |
| 6 | No duplicate check | nb 01/05 | Near-identical images can sit in train and test | pHash + dHash de-duplication, group-aware folds |
| 7 | "XGBoost / LightGBM / CatBoost" were the same algorithm with different lambda/gamma | nb 10, cell 4 | Misleading naming in a paper | Renamed GBDT-A/B/C |
| 8 | Dataset folder name `fabricfaultdetection` used as a class label in `metadata.csv` | nb 01 | A 7th "class" that is a data source, not a defect | Explicit alias table; unknown folders are reported and skipped |
| 9 | Class names lost in deployment: `idx_to_class` maps ints to ints (`np.int64(0) -> 0`) | nb 06/11 | App shows "Result: 0" instead of a class name | `model_config.json` stores the class names |
| 10 | Deployment loads a pickle that needs `__main__` classes | `deployment/inference.py` | Fragile, and the Hub flags `deployment_model.pkl` as unsafe | `model.safetensors` + vendored model code |
| 11 | Hard-coded Colab/Drive paths everywhere | all notebooks | Not reproducible outside one Drive | Package + CLI scripts + YAML configs |
| 12 | Grid search with one combination per model | nb 10, cell 6 | It is not a search | Fixed recipe shared by all models, documented in `configs/` |
| 14 | 75% of the images in the defect folders are `N_processed (k).jpg`: thresholded, binarised copies of `N.jpg` (3 per photo) | dataset | A random split puts a photo in train and its binarised twin in test. This is the largest source of inflated accuracy | Copies are grouped with their source (same fold), used for training only, never for validation or test |
| 15 | `exp*_num_*.png` (and `- Copy` duplicates) in *Vertical* are binary line masks, not fabric photos | dataset | Not the task; trivially separable | Dropped as non-photographs |
| 16 | Files named `line_...` inside the *hole* folder | dataset | Possible label noise | Listed in `data/name_label_conflicts.csv` for manual checking |
| 17 | Defect-free images are mostly 4096x256 grey line-scan strips; defects are 640x360 colour photos and 12 MP phone photos | dataset | A model can tell classes apart by size, colour and distortion after resizing (source shortcut) | Elongated images centre-cropped to squares, all input converted to greyscale, `scripts/shortcut_check.py` measures what is left |
| 13 | `shap_results/`, `results/`, `webapp/` folders empty | repo | Explainability claimed but not done | Grad-CAM, spectral saliency, SGA gate figures |

## Why v1 numbers should not be compared with v2 numbers directly

v1 used one split without de-duplication and reported accuracy. v2 re-runs the v1 method
(`scripts/run_legacy_v1.py`) twice: on a split that mimics v1 on purpose (`--prefix leaky_`,
copies treated as independent, no grouping, colour) and on the v2 folds. The difference between the
two is the measured effect of leakage, and the paper reports it.
Expect the v1 number to drop under the new protocol. That drop is a finding worth reporting,
not something to hide.
