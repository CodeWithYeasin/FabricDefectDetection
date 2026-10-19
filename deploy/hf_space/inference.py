"""Pickle-free inference for the Hugging Face Space.

Weights are loaded from ``model.safetensors`` (tensors only, no arbitrary code execution) and the
architecture is rebuilt from ``model_config.json`` + the vendored ``fabricnet_models`` package.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from fabricnet_models import SpecTexNet, build_model, gradcam_target_layer

HERE = Path(__file__).parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FabricPredictor:
    def __init__(self, folder: str | Path = HERE, device: str | None = None):
        folder = Path(folder)
        self.meta = json.loads((folder / "model_config.json").read_text())
        self.classes = self.meta["classes"]
        self.size = self.meta["img_size"]
        self.grayscale = self.meta.get("grayscale", False)
        m = dict(self.meta["model"])
        name = m.pop("name")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(name, len(self.classes), pretrained=False, **m)
        self.model.load_state_dict(load_file(folder / "model.safetensors"))
        self.model.to(self.device).eval()
        h, v = self.classes.index("horizontal"), self.classes.index("vertical")
        self.perm = list(range(len(self.classes)))
        self.perm[h], self.perm[v] = v, h

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        img = (img.convert("L") if self.grayscale else img).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1))[None].to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image, tta: bool = True) -> dict[str, float]:
        x = self.preprocess(img)
        views = [(x, False)]
        if tta:
            views += [(torch.flip(x, [-1]), False), (torch.flip(x, [-2]), False),
                      (torch.rot90(x, 1, [-2, -1]), True), (torch.rot90(x, 3, [-2, -1]), True)]
        ps = []
        for v, swap in views:
            p = torch.softmax(self.model(v), -1)[0]
            ps.append(p[self.perm] if swap else p)
        p = torch.stack(ps).mean(0).cpu().numpy()
        return {c: float(p[i]) for i, c in enumerate(self.classes)}

    def heatmap(self, img: Image.Image, target: int | None = None) -> np.ndarray:
        """Grad-CAM over the last backbone stage, in [0,1], at model resolution."""
        layer = gradcam_target_layer(self.model)
        store = {}

        def hook(_m, _i, out):
            out = out[-1] if isinstance(out, (list, tuple)) else out
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))

        h = layer.register_forward_hook(hook)
        try:
            x = self.preprocess(img).requires_grad_(True)
            logits = self.model(x)
            t = int(logits.argmax(1)) if target is None else target
            self.model.zero_grad(set_to_none=True)
            logits[0, t].backward()
            w = store["g"].mean((-2, -1), keepdim=True)
            cam = F.relu((w * store["a"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(self.size, self.size), mode="bilinear", align_corners=False)[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.detach().cpu().numpy()
        finally:
            h.remove()

    @torch.no_grad()
    def spectral_map(self, img: Image.Image) -> np.ndarray | None:
        if not isinstance(self.model, SpecTexNet) or self.model.fusion == "none":
            return None
        s = self.model.spec(self.preprocess(img))[:, -1:]
        s = F.interpolate(s, size=(self.size, self.size), mode="bilinear", align_corners=False)[0, 0]
        return ((s - s.min()) / (s.max() - s.min() + 1e-8)).cpu().numpy()
