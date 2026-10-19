#!/usr/bin/env python
"""Run a list of experiments on all visible GPUs in parallel (one experiment per GPU at a time).

Kaggle's "GPU T4 x2" gives two GPUs; scripts/train.py uses one, so this halves wall-clock time.
Finished folds are skipped, and no new fold starts after --hours, so a 12-hour session ends cleanly.

python scripts/run_queue.py --hours 11 --exps spectex_convnext_tiny convnext_tiny abl_no_spectral
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def n_gpus() -> int:
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=11.0, help="stop starting new folds after this many hours")
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--workers-per-job", type=int, default=2, help="dataloader workers per experiment")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    deadline = time.time() + a.hours * 3600
    gpus = a.gpus or n_gpus()
    q: queue.Queue = queue.Queue()
    for e in a.exps:
        q.put(e)
    logs = ROOT / "results" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = {}

    def worker(gpu: int):
        while time.time() < deadline:
            try:
                e = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--exp", e, "--skip-done",
                   "--deadline", str(deadline), "--set", f"data.num_workers={a.workers_per_job}", *a.set]
            print(f"[gpu{gpu}] start {e}", flush=True)
            t0 = time.time()
            with open(logs / f"{e}.log", "a") as f:
                rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            status[e] = rc
            print(f"[gpu{gpu}] done  {e} rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    left = []
    while not q.empty():
        left.append(q.get())
    failed = [e for e, rc in status.items() if rc != 0]
    print(f"ran (complete, or stopped at the time budget): {sorted(k for k, v in status.items() if v == 0)}")
    if failed:
        print(f"FAILED (see results/_logs/<exp>.log): {failed}")
    if left:
        print(f"not started (time budget): {left}. Run the same cell again in the next session.")


if __name__ == "__main__":
    main()
