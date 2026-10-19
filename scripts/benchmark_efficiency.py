#!/usr/bin/env python
"""Params / GMACs / latency for every model in the main comparison -> results/efficiency.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from fabricnet.efficiency import count_gflops, count_params, latency_ms  # noqa: E402
from fabricnet.models import build_model  # noqa: E402
from fabricnet.utils import get_logger, save_json  # noqa: E402
from train import experiment_config  # noqa: E402

log = get_logger()
DEFAULT = ["mobilenetv3_large_100", "efficientnet_b0", "resnet50", "convnext_tiny", "swin_tiny", "deit_small",
           "spectex_efficientnet_b0", "spectex_resnet50", "spectex_convnext_tiny"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=DEFAULT)
    ap.add_argument("--threads", type=int, default=1, help="1 = single-core edge setting; also avoids MKL-FFT thread spin-up overhead")
    ap.add_argument("--out", default="results/efficiency.json")
    a = ap.parse_args()
    out = {}
    for e in a.exps:
        cfg = experiment_config(e)
        m = cfg["model"]
        model = build_model(m["name"], len(cfg["classes"]), pretrained=False,
                            **{k: v for k, v in m.items() if k not in {"name", "pretrained"}})
        r = {"params_m": count_params(model), **count_gflops(model, cfg["data"]["img_size"]),
             "cpu": latency_ms(model, cfg["data"]["img_size"], "cpu", threads=a.threads)}
        if torch.cuda.is_available():
            r["gpu"] = latency_ms(model, cfg["data"]["img_size"], "cuda", batch=32)
        out[e] = r
        log.info(f"{e}: {r['params_m']:.2f}M params, {r['total_gmacs']:.2f} GMACs, CPU {r['cpu']['median_ms']:.1f} ms")
    save_json(out, a.out)


if __name__ == "__main__":
    main()
