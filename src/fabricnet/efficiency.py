"""Parameter count, FLOPs and latency.

FLOPs come from ``torch.utils.flop_counter`` (counts matmul/conv; FFTs are not counted, so we
add an analytic 5·N·log2(N) per complex 2-D FFT for SpecTexNet and report it separately).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .models import SpecTexNet


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gflops(model, img_size: int = 224) -> dict:
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    gf = fc.get_total_flops() / 1e9 / 2  # report multiply-adds (GMACs) like most papers
    fft = 0.0
    if isinstance(model, SpecTexNet) and model.fusion != "none":
        for s in model.spec.scales:
            n = s * s
            fft += 2 * 5 * n * math.log2(n) / 1e9 / 2  # forward + inverse FFT
    return {"gmacs": gf, "fft_gmacs": fft, "total_gmacs": gf + fft}


@torch.no_grad()
def latency_ms(model, img_size: int = 224, device: str = "cpu", batch: int = 1, warmup: int = 10,
               runs: int = 50, threads: int | None = None) -> dict:
    if threads and device == "cpu":
        torch.set_num_threads(threads)
    model = model.eval().to(device)
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
            "fps": float(batch * 1000 / np.median(ts)), "device": device, "batch": batch}
