#!/usr/bin/env python
"""Aggregate all finished experiments into summary JSON, significance tests, LaTeX tables and figures.

python scripts/aggregate.py --proposed spectex_convnext_tiny

Reads   results/<exp>/r*_f*/{metrics.json,test_predictions*.csv}, data/dedup_report.json,
        results/efficiency.json (optional), results/cross_dataset_*.json (optional)
Writes  results/summary.json, results/stats.json,
        paper/tables/*.tex, paper/tables/results_macros.tex, paper/figures/*.pdf
Tables only contain experiments that actually ran; nothing is ever filled in by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef  # noqa: E402

from fabricnet.metrics import HEADLINE, compute_metrics  # noqa: E402
from fabricnet.stats import bootstrap_ci, friedman_nemenyi, mcnemar, plot_cd_diagram, wilcoxon_holm  # noqa: E402
from fabricnet.utils import get_logger, load_config, load_json, save_json  # noqa: E402

log = get_logger()

DISPLAY = {
    "v1_frozen_r50_linear": "Frozen ResNet50 + linear probe",
    "v1_frozen_r50_gbdt_a": "Frozen ResNet50 + GBDT (v1)",
    "v1_frozen_r50_gbdt_ens": "Frozen ResNet50 + GBDT ensemble (v1)",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv3_large_100": "MobileNetV3-L",
    "convnext_tiny": "ConvNeXt-T",
    "swin_tiny": "Swin-T",
    "deit_small": "DeiT-S",
    "spectex_resnet50": "SpecTexNet (ResNet50)",
    "spectex_efficientnet_b0": "SpecTexNet (EffNet-B0)",
    "spectex_convnext_tiny": "SpecTexNet (ConvNeXt-T)",
    "abl_no_spectral": "w/o spectral branch",
    "abl_concat_fusion": "SGA $\\rightarrow$ late concat",
    "abl_fixed_sr": "learnable SR $\\rightarrow$ fixed SR",
    "abl_single_scale": "multi-scale $\\rightarrow$ single scale (64)",
    "abl_aug_flip": "OA-Rot $\\rightarrow$ flips only",
    "abl_aug_naive_rot": "OA-Rot $\\rightarrow$ naive rotation",
    "abl_loss_ce": "CB-Focal $\\rightarrow$ CE",
    "abl_no_derived": "w/o binarised copies in training",
    "abl_colour": "greyscale $\\rightarrow$ RGB input",
    "__notta": "w/o OA-TTA",
}
MAIN_ORDER = ["v1_frozen_r50_linear", "v1_frozen_r50_gbdt_a", "v1_frozen_r50_gbdt_ens", "mobilenetv3_large_100",
              "efficientnet_b0", "resnet50", "convnext_tiny", "swin_tiny", "deit_small",
              "spectex_efficientnet_b0", "spectex_resnet50", "spectex_convnext_tiny"]
ABL_ORDER = ["abl_no_spectral", "abl_concat_fusion", "abl_fixed_sr", "abl_single_scale",
             "abl_aug_flip", "abl_aug_naive_rot", "abl_loss_ce", "abl_no_derived", "abl_colour", "__notta"]
TABLE_METRICS = [("accuracy", "Acc"), ("balanced_accuracy", "BAcc"), ("macro_f1", "Macro-F1"),
                 ("mcc", "MCC"), ("macro_auroc", "AUROC")]


def load_exp(d: Path, classes, notta=False):
    runs = sorted(d.glob("r*_f*"))
    fname = "test_predictions_notta.csv" if notta else "test_predictions.csv"
    per_run, preds = {}, []
    for r in runs:
        f = r / fname
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p["run"] = r.name
        preds.append(p)
        per_run[r.name] = compute_metrics(p["y"].values, p[[f"p_{c}" for c in classes]].values, classes)
    if not preds:
        return None
    return {"per_run": per_run, "preds": pd.concat(preds, ignore_index=True)}


def summarise(e, classes, n_boot):
    keys = [k for k in HEADLINE if all(k in m for m in e["per_run"].values())]
    agg = {k: {"mean": float(np.mean([m[k] for m in e["per_run"].values()])),
               "std": float(np.std([m[k] for m in e["per_run"].values()], ddof=1)) if len(e["per_run"]) > 1 else 0.0}
           for k in keys}
    r0 = e["preds"][e["preds"]["run"].str.startswith("r0_")]
    y, P = r0["y"].values, r0[[f"p_{c}" for c in classes]].values
    pooled = compute_metrics(y, P, classes)
    ci = {}
    for name, fn in {"macro_f1": lambda a, b: f1_score(a, b.argmax(1), average="macro", labels=range(len(classes)), zero_division=0),
                     "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b.argmax(1)),
                     "mcc": lambda a, b: matthews_corrcoef(a, b.argmax(1)),
                     "accuracy": lambda a, b: float((a == b.argmax(1)).mean())}.items():
        ci[name] = bootstrap_ci(y, P, fn, n_boot=n_boot)
    return {"n_runs": len(e["per_run"]), "cv": agg, "pooled": pooled, "ci95": ci}


def pct(x):
    return f"{100 * x:.2f}"


def fmt_cell(s, k, best):
    cv = s["cv"].get(k)
    if cv is None:
        return "--"
    txt = f"{pct(cv['mean'])}$\\pm${pct(cv['std'])}" if k != "mcc" else f"{cv['mean']:.3f}$\\pm${cv['std']:.3f}"
    return f"\\textbf{{{txt}}}" if best else txt


def table(rows, summary, caption, label, path, metrics=TABLE_METRICS, highlight=None):
    rows = [r for r in rows if r in summary]
    if not rows:
        path.write_text(f"% {label}: no finished experiments yet\n")
        return
    best = {k: max(rows, key=lambda r: summary[r]["cv"].get(k, {"mean": -1})["mean"]) for k, _ in metrics}
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{l" + "c" * len(metrics) + "}",
             "\\toprule", "Method & " + " & ".join(h for _, h in metrics) + " \\\\", "\\midrule"]
    for r in rows:
        name = DISPLAY.get(r, r.replace("_", "\\_"))
        if r == highlight:
            name = f"\\textbf{{{name}}} (ours)"
        lines.append(name + " & " + " & ".join(fmt_cell(summary[r], k, best[k] == r) for k, _ in metrics) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def ablation_table(summary, proposed, path):
    if proposed not in summary:
        path.write_text("% ablation: proposed model not finished yet\n")
        return
    ref = summary[proposed]["cv"]
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Ablation study (5-fold CV, mean$\\pm$std). "
             "$\\Delta$ is the change in macro-F1 (percentage points) relative to the full model.}",
             "\\label{tab:ablation}", "\\small", "\\begin{tabular}{lccc}", "\\toprule",
             "Variant & Macro-F1 & BAcc & $\\Delta$F1 \\\\", "\\midrule",
             f"Full model & {fmt_cell(summary[proposed], 'macro_f1', False)} & "
             f"{fmt_cell(summary[proposed], 'balanced_accuracy', False)} & -- \\\\"]
    for a in ABL_ORDER:
        key = proposed + "__notta" if a == "__notta" else a
        if key not in summary:
            continue
        d = 100 * (summary[key]["cv"]["macro_f1"]["mean"] - ref["macro_f1"]["mean"])
        lines.append(f"{DISPLAY[a]} & {fmt_cell(summary[key], 'macro_f1', False)} & "
                     f"{fmt_cell(summary[key], 'balanced_accuracy', False)} & {d:+.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def perclass_table(summary, rows, classes, path):
    rows = [r for r in rows if r in summary]
    if not rows:
        path.write_text("% per-class: nothing yet\n")
        return
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Per-class F1 (\\%) on pooled out-of-fold predictions.}",
             "\\label{tab:perclass}", "\\small", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "c" * len(classes) + "}", "\\toprule",
             "Method & " + " & ".join(c.replace("_", " ") for c in classes) + " \\\\", "\\midrule"]
    for r in rows:
        pc = summary[r]["pooled"]["per_class"]
        lines.append(DISPLAY.get(r, r.replace("_", "\\_")) + " & " + " & ".join(pct(pc[c]["f1"]) for c in classes) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def dataset_table(report, classes, path):
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Dataset composition before and after duplicate removal.}",
             "\\label{tab:dataset}", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
             "Class & Raw files & Independent images & Binarised copies (train only) \\\\", "\\midrule"]
    der = report.get("train_only_derived_counts", {})
    for c in classes:
        lines.append(f"{c.replace('_', ' ')} & {report['raw_class_counts'].get(c, 0)} & "
                     f"{report['clean_class_counts'].get(c, 0)} & {der.get(c, 0)} \\\\")
    lines += ["\\midrule", f"Total & {sum(report['raw_class_counts'].values())} & {report['n_after']} & "
              f"{sum(der.values())} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def efficiency_table(eff, summary, path):
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Model complexity and CPU latency (batch 1, 224$\\times$224).}",
             "\\label{tab:efficiency}", "\\small", "\\begin{tabular}{lrrrc}", "\\toprule",
             "Model & Params (M) & GMACs & CPU ms & Macro-F1 \\\\", "\\midrule"]
    for k, v in eff.items():
        f1 = pct(summary[k]["cv"]["macro_f1"]["mean"]) if k in summary else "--"
        lines.append(f"{DISPLAY.get(k, k.replace('_', chr(92) + '_'))} & {v['params_m']:.2f} & {v['total_gmacs']:.2f} & "
                     f"{v['cpu']['median_ms']:.1f} & {f1} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def macros(summary, stats, proposed, report, path):
    out = ["% auto-generated by scripts/aggregate.py -- do not edit"]

    def m(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    s = summary.get(proposed)
    if s:
        m("PropMacroFone", pct(s["cv"]["macro_f1"]["mean"]))
        m("PropMacroFoneStd", pct(s["cv"]["macro_f1"]["std"]))
        m("PropBAcc", pct(s["cv"]["balanced_accuracy"]["mean"]))
        m("PropAcc", pct(s["cv"]["accuracy"]["mean"]))
        m("PropMCC", f"{s['cv']['mcc']['mean']:.3f}")
        lo, hi = s["ci95"]["macro_f1"][1:]
        m("PropMacroFoneCI", f"[{pct(lo)}, {pct(hi)}]")
        if "defect_detection_rate" in s["cv"]:
            m("PropDDR", pct(s["cv"]["defect_detection_rate"]["mean"]))
            m("PropFAR", pct(s["cv"]["false_alarm_rate"]["mean"]))
    v1 = summary.get("v1_frozen_r50_gbdt_ens")
    if v1:
        m("VoneMacroFone", pct(v1["cv"]["macro_f1"]["mean"]))
        m("VoneAcc", pct(v1["cv"]["accuracy"]["mean"]))
    if s and v1:
        m("GainOverVone", f"{100 * (s['cv']['macro_f1']['mean'] - v1['cv']['macro_f1']['mean']):.2f}")
    if report:
        m("NRaw", sum(report["raw_class_counts"].values()))
        m("NClean", report["n_after"])
        m("NDupGroups", report["duplicate_groups"])
        m("NDupImages", report["images_in_duplicate_groups"])
        m("NDerived", report.get("derived_copies", 0))
        m("NNonPhoto", report.get("nonphoto_dropped", 0))
        m("NNameConflicts", report.get("name_label_conflicts", 0))
        m("NCropped", report.get("center_cropped_elongated", 0))
    sc = stats.get("_shortcut")
    if sc:
        m("ShortcutMetaFone", pct(sc["metadata_only"]["macro_f1"]))
        m("ShortcutPixFone", pct(sc["processed_input_stats"]["macro_f1"]))
    lk = summary.get("leaky_v1_frozen_r50_gbdt_ens")
    if lk:
        m("LeakyVoneAcc", pct(lk["cv"]["accuracy"]["mean"]))
        m("LeakyVoneMacroFone", pct(lk["cv"]["macro_f1"]["mean"]))
    if stats.get("friedman"):
        m("FriedmanP", f"{stats['friedman']['p_value']:.2g}")
        m("NemenyiCD", f"{stats['friedman']['nemenyi_cd']:.2f}")
    path.write_text("\n".join(out) + "\n")


def figures(summary, preds, proposed, classes, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if proposed in summary:
        cm = np.array(summary[proposed]["pooled"]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        lab = [c.replace("_", " ") for c in classes]
        ax.set_xticks(range(len(classes)), lab, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)), lab, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_proposed.pdf")
        plt.close(fig)

        p = preds[proposed]
        p = p[p["run"].str.startswith("r0_")]
        P = p[[f"p_{c}" for c in classes]].values
        conf, acc = P.max(1), (P.argmax(1) == p["y"].values)
        bins = np.linspace(0, 1, 11)
        mids, accs, cnt = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.any():
                mids.append((lo + hi) / 2)
                accs.append(acc[mk].mean())
                cnt.append(mk.sum())
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax.bar(mids, accs, width=0.09, edgecolor="k", lw=0.5, color="C0", alpha=0.8)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"ECE = {summary[proposed]['pooled']['ece']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "reliability_proposed.pdf")
        plt.close(fig)

    rows = [r for r in MAIN_ORDER if r in summary]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        w = 0.8 / len(rows)
        for i, r in enumerate(rows):
            f1s = [summary[r]["pooled"]["per_class"][c]["f1"] for c in classes]
            ax.bar(np.arange(len(classes)) + i * w, f1s, w, label=DISPLAY.get(r, r).replace("$\\rightarrow$", "->"))
        ax.set_xticks(np.arange(len(classes)) + 0.4 - w / 2, [c.replace("_", " ") for c in classes], fontsize=8)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=5.5, ncol=3, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / "perclass_f1.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--proposed", default="spectex_convnext_tiny")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--paper-dir", default=str(ROOT / "paper"))
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config, args.set)
    classes = cfg["classes"]
    res_root = Path(cfg["output_root"])
    tab_dir, fig_dir = Path(args.paper_dir) / "tables", Path(args.paper_dir) / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary, preds, per_fold = {}, {}, {}
    for d in sorted(p for p in res_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for notta in (False, True):
            e = load_exp(d, classes, notta)
            if e is None:
                continue
            key = d.name + ("__notta" if notta else "")
            summary[key] = summarise(e, classes, args.n_boot)
            preds[key] = e["preds"]
            per_fold[key] = {r: m["macro_f1"] for r, m in e["per_run"].items()}
            log.info(f"{key}: macro-F1 {summary[key]['cv']['macro_f1']['mean']:.4f} "
                     f"± {summary[key]['cv']['macro_f1']['std']:.4f} ({summary[key]['n_runs']} runs)")
    if not summary:
        raise SystemExit(f"no finished runs under {res_root}")
    save_json(summary, res_root / "summary.json")

    # ---- significance (main comparison) ----
    stats = {}
    main_rows = [r for r in MAIN_ORDER if r in summary]
    common = sorted(set.intersection(*[set(per_fold[r]) for r in main_rows])) if main_rows else []
    if len(main_rows) >= 3 and len(common) >= 2:
        S = np.array([[per_fold[r][run] for r in main_rows] for run in common])
        stats["friedman"] = friedman_nemenyi(S, main_rows)
        plot_cd_diagram({DISPLAY.get(k, k).replace("$\\rightarrow$", "->"): v
                         for k, v in stats["friedman"]["avg_rank"].items()},
                        stats["friedman"]["nemenyi_cd"], str(fig_dir / "cd_diagram.pdf"), "Macro-F1 ranks over folds")
        if args.proposed in main_rows:
            stats["wilcoxon_holm"] = wilcoxon_holm(S, main_rows, args.proposed)
    if args.proposed in preds:
        pa = preds[args.proposed]
        pa = pa[pa["run"].str.startswith("r0_")].set_index("path")
        stats["mcnemar"] = {}
        for r in summary:
            if r == args.proposed or r.endswith("__notta") or r.startswith("leaky_"):
                continue
            pb = preds[r]
            pb = pb[pb["run"].str.startswith("r0_")].set_index("path")
            idx = pa.index.intersection(pb.index)
            stats["mcnemar"][r] = mcnemar(pa.loc[idx, "y"].values, pa.loc[idx, "pred"].values, pb.loc[idx, "pred"].values)
    save_json(stats, res_root / "stats.json")

    # ---- tables & figures ----
    report_p = ROOT / Path(cfg["data"]["splits_csv"]).parent / "dedup_report.json"
    report = load_json(report_p) if report_p.exists() else None
    table(MAIN_ORDER, summary,
          "Comparison with baselines under the leakage-free 5-fold protocol (mean$\\pm$std over folds, \\%). "
          "All fine-tuned models share the same training recipe.", "tab:main", tab_dir / "table_main.tex",
          highlight=args.proposed)
    ablation_table(summary, args.proposed, tab_dir / "table_ablation.tex")
    perclass_table(summary, [r for r in MAIN_ORDER if r in summary], classes, tab_dir / "table_perclass.tex")
    if report:
        dataset_table(report, classes, tab_dir / "table_dataset.tex")
    eff_p = res_root / "efficiency.json"
    if eff_p.exists():
        efficiency_table(load_json(eff_p), summary, tab_dir / "table_efficiency.tex")
    if stats.get("mcnemar"):
        lines = ["\\begin{table}[t]", "\\centering",
                 f"\\caption{{McNemar tests: {DISPLAY.get(args.proposed, args.proposed)} vs.\\ each model on pooled "
                 "out-of-fold predictions. $n_{01}$: only ours correct; $n_{10}$: only the other correct.}",
                 "\\label{tab:mcnemar}", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
                 "Compared model & $n_{01}$ & $n_{10}$ & $p$ \\\\", "\\midrule"]
        for r in [x for x in MAIN_ORDER + ABL_ORDER if x in stats["mcnemar"]]:
            v = stats["mcnemar"][r]
            lines.append(f"{DISPLAY.get(r, r)} & {v['a_only_correct']} & {v['b_only_correct']} & {v['p_value']:.3g} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (tab_dir / "table_mcnemar.tex").write_text("\n".join(lines) + "\n")
    sc_p = res_root / "shortcut_check.json"
    if sc_p.exists():
        stats["_shortcut"] = load_json(sc_p)
    macros(summary, stats, args.proposed, report, tab_dir / "results_macros.tex")
    figures(summary, preds, args.proposed, classes, fig_dir)
    log.info(f"tables -> {tab_dir}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
