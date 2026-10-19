#!/usr/bin/env python
"""Export a trained run to the Hugging Face Space folder as safetensors (no pickle).

python scripts/export_model.py --run results/spectex_convnext_tiny/full
python scripts/export_model.py --run results/spectex_convnext_tiny/r0_f0     # if you skipped --full

Writes deploy/hf_space/{model.safetensors, model_config.json} and copies the model source files
into deploy/hf_space/fabricnet_models/ so the Space does not depend on this repo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--space", default=str(ROOT / "deploy/hf_space"))
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / "best.pt", map_location="cpu", weights_only=False)
    space = Path(a.space)
    sd = {k: v.contiguous() for k, v in ck["model"].items()}
    save_file(sd, space / "model.safetensors")
    m = ck["cfg"]["model"]
    meta = {"model": {k: v for k, v in m.items() if k != "pretrained"}, "classes": ck["classes"],
            "img_size": ck["cfg"]["data"]["img_size"],
            "grayscale": ck["cfg"]["data"].get("grayscale", False), "tta": ck["cfg"].get("eval", {}).get("tta", "none"),
            "source_run": str(a.run), "epoch": ck.get("epoch")}
    (space / "model_config.json").write_text(json.dumps(meta, indent=2))
    sync_model_sources(space)
    print(f"exported to {space}")


def sync_model_sources(space: Path):
    dst = space / "fabricnet_models"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["spectral.py", "spectexnet.py", "factory.py"]:
        shutil.copy(ROOT / "src/fabricnet/models" / f, dst / f)
    (dst / "__init__.py").write_text(
        "from .factory import TimmClassifier, build_model, gradcam_target_layer\n"
        "from .spectexnet import SpecTexNet\n"
        "from .spectral import SpectralResidual\n")


if __name__ == "__main__":
    main()
