"""Learnable multi-scale Spectral Residual (LSR) saliency.

Background
----------
Woven and knitted fabric is close to periodic, so its 2-D Fourier log-amplitude spectrum is
dominated by sharp peaks at the weave frequencies on top of a smooth envelope. A defect breaks
the periodicity and shows up as the *non-smooth remainder* of the spectrum. Hou & Zhang (CVPR
2007) called this remainder the spectral residual:

    L(f) = log |F(f)|,     R(f) = L(f) - (h * L)(f),     S = | F^-1[ exp(R + i*phase) ] |^2

where h is a fixed 3x3 mean filter. We make two changes:

1. ``h`` is a learnable, softmax-normalised kernel per scale (initialised to the mean filter), so
   the network can tune how much of the periodic background it suppresses.
2. The residual is computed at several resolutions (default 64 and 128 px) because weave pitch
   relative to image size varies across datasets and cameras.

The output is a stack of saliency maps (one per scale) that a small CNN then encodes.
Everything is differentiable and runs on GPU via ``torch.fft``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


class SpectralResidual(nn.Module):
    def __init__(self, scales=(64, 128), kernel_size: int = 3, learnable: bool = True,
                 blur_sigma: float = 1.5, out_size: int = 112):
        super().__init__()
        self.scales = tuple(scales)
        self.k = kernel_size
        self.out_size = out_size
        self.learnable = learnable
        # logits for a softmax-normalised smoothing kernel; zeros => uniform mean filter
        init = torch.zeros(len(self.scales), 1, kernel_size * kernel_size)
        self.kernel_logits = nn.Parameter(init, requires_grad=learnable)
        self.register_buffer("gauss", _gaussian_kernel(blur_sigma), persistent=False)

    @property
    def n_out(self) -> int:
        return len(self.scales)

    def _smooth_kernel(self, s: int) -> torch.Tensor:
        w = torch.softmax(self.kernel_logits[s], dim=-1)
        return w.view(1, 1, self.k, self.k)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gauss.to(x.dtype)
        r = g.numel() // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))
        return x

    def saliency(self, gray: torch.Tensor, s: int) -> torch.Tensor:
        size = self.scales[s]
        g = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        g = g - g.mean(dim=(-2, -1), keepdim=True)
        # symmetric padding removes the wrap-around discontinuity that otherwise lights up borders
        p = size // 8
        g = F.pad(g, (p, p, p, p), mode="reflect")
        spec = torch.fft.fft2(g.float())
        amp, phase = spec.abs(), torch.angle(spec)
        log_amp = torch.log(amp + 1e-8)
        # smooth the *centred* spectrum (neighbouring frequencies), then un-centre
        la = torch.fft.fftshift(log_amp, dim=(-2, -1))
        pad = self.k // 2
        smooth = F.conv2d(F.pad(la, (pad, pad, pad, pad), mode="replicate"), self._smooth_kernel(s))
        resid = torch.fft.ifftshift(la - smooth, dim=(-2, -1))
        sal = torch.fft.ifft2(torch.exp(resid) * torch.exp(1j * phase)).abs() ** 2
        sal = self._blur(sal)[..., p:-p, p:-p]
        sal = F.interpolate(sal, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        # per-image standardisation in log domain (saliency is heavy-tailed)
        sal = torch.log1p(sal / (sal.mean(dim=(-2, -1), keepdim=True) + 1e-8))
        mu = sal.mean(dim=(-2, -1), keepdim=True)
        sd = sal.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (sal - mu) / sd

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: ImageNet-normalised RGB (B,3,H,W) -> (B, n_scales, out, out)."""
        x = x_norm * _STD.to(x_norm) + _MEAN.to(x_norm)
        gray = (x * _GRAY.to(x)).sum(1, keepdim=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            maps = [self.saliency(gray.float(), s) for s in range(len(self.scales))]
        return torch.cat(maps, dim=1).to(x_norm.dtype)
