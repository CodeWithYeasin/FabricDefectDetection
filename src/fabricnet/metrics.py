"""Metrics suited to an imbalanced multi-class defect problem.

Plain accuracy is reported for continuity with v1 but is NOT the headline metric: with ~60%
defect-free images a trivial classifier already scores 60%. Headline = macro-F1, plus balanced
accuracy and MCC. Industrial metrics (defect detection rate / false alarm rate) collapse the
problem to defect vs. defect-free.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, probs: np.ndarray, classes: list[str], normal_class: str = "defect_free") -> dict:
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    K = len(classes)
    pred = probs.argmax(1)
    labels = list(range(K))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "ece": expected_calibration_error(probs, y),
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)))),
    }
    present = np.unique(y)
    if len(present) == K:
        out["macro_auroc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels)
        Y = np.eye(K)[y]
        out["macro_auprc"] = average_precision_score(Y, probs, average="macro")
    out["per_class"] = {
        c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(classes)
    }
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=labels).tolist()
    if normal_class in classes:
        n = classes.index(normal_class)
        true_def, pred_def = y != n, pred != n
        out["defect_detection_rate"] = float((pred_def & true_def).sum() / max(true_def.sum(), 1))
        out["false_alarm_rate"] = float((pred_def & ~true_def).sum() / max((~true_def).sum(), 1))
        p_def = 1 - probs[:, n]
        if 0 < true_def.sum() < len(y):
            out["binary_auroc"] = roc_auc_score(true_def, p_def)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


HEADLINE = ["macro_f1", "balanced_accuracy", "mcc", "accuracy", "macro_auroc", "ece",
            "defect_detection_rate", "false_alarm_rate"]
