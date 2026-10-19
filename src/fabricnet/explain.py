"""Grad-CAM (Selvaraju et al., ICCV 2017) and SpecTexNet attention / spectral-saliency maps."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import SpecTexNet, gradcam_target_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module | None = None):
        self.model = model.eval()
        self.layer = layer or gradcam_target_layer(model)
        self.acts = self.grads = None
        self.h1 = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.dim() == 4 and out.shape[-1] > out.shape[1] and out.shape[1] == out.shape[2]:
            out = out.permute(0, 3, 1, 2)  # NHWC -> NCHW (swin-style)
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
        x = x.requires_grad_(True)
        logits = self.model(x)
        if target is None:
            target = int(logits.argmax(1)[0])
        self.model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        w = self.grads.mean((-2, -1), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), target

    def remove(self):
        self.h1.remove()


@torch.no_grad()
def spectral_maps(model: SpecTexNet, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Spectral saliency per scale + SGA spatial gate, all upsampled to input size, in [0,1]."""
    model.eval()
    out = {}
    if model.fusion == "none":
        return out
    sal = model.spec(x)
    for i, s in enumerate(model.spec.scales):
        m = F.interpolate(sal[:, i : i + 1], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        out[f"spectral_{s}"] = m.cpu().numpy()
    model(x)
    if model._last_attn is not None:
        a = F.interpolate(model._last_attn, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        out["sga_gate"] = ((a - a.min()) / (a.max() - a.min() + 1e-8)).cpu().numpy()
    return out


def overlay(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib

    h = matplotlib.colormaps["jet"](heat)[..., :3]
    base = img_rgb.astype(np.float32) / 255.0
    return (np.clip((1 - alpha) * base + alpha * h, 0, 1) * 255).astype(np.uint8)
