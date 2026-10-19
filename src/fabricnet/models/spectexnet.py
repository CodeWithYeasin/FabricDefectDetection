"""SpecTexNet: a dual-branch network that fuses CNN appearance features with spectral-residual
"periodicity-break" cues through Spectral-Guided Attention (SGA).

        RGB ──► CNN backbone ───────────────► F (B,C,h,w) ──┐
         │                                                 ├─► SGA ─► GAP(F') ⊕ GAP(G) ─► FC
         └──► Learnable Spectral Residual ─► small CNN ─► G (B,c,h,w)

SGA (fusion="sga"):
    a  = σ( Conv1x1([ReduceC(F) ; G]) )            spatial gate (B,1,h,w)
    s  = σ( MLP(GAP(G)) )                          channel gate (B,C)
    F' = F ⊙ (1 + a) ⊙ (1 + s)                     residual gating; F is never zeroed out

Ablation switches:
    fusion = "sga" | "concat" (late concat only) | "none" (backbone only, spectral branch off)
    spectral.learnable = True | False (fixed Hou&Zhang mean filter)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .spectral import SpectralResidual


def _conv_bn(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class SpectralEncoder(nn.Module):
    """Light CNN over the saliency maps. 112 -> 7 with four stride-2 stages (~0.4M params)."""

    def __init__(self, cin: int, width: int = 32, cout: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn(cin, width, 2),
            _conv_bn(width, width * 2, 2),
            _conv_bn(width * 2, width * 4, 2),
            _conv_bn(width * 4, cout, 2),
        )

    def forward(self, x):
        return self.net(x)


class SpecTexNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", num_classes: int = 6, pretrained: bool = True,
                 fusion: str = "sga", spec_scales=(64, 128), spec_learnable: bool = True,
                 spec_channels: int = 128, drop_rate: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()
        assert fusion in {"sga", "concat", "none"}
        self.fusion = fusion
        kw = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                              out_indices=(-1,), **kw)
        except TypeError:  # some archs don't take drop_path_rate
            self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(-1,))
        C = self.backbone.feature_info.channels()[-1]
        self.C = C
        feat_dim = C
        if fusion != "none":
            self.spec = SpectralResidual(scales=spec_scales, learnable=spec_learnable)
            self.spec_enc = SpectralEncoder(self.spec.n_out, cout=spec_channels)
            feat_dim += spec_channels
        if fusion == "sga":
            r = 128
            self.reduce = nn.Sequential(nn.Conv2d(C, r, 1, bias=False), nn.BatchNorm2d(r), nn.SiLU(inplace=True))
            self.spatial_gate = nn.Conv2d(r + spec_channels, 1, 1)
            self.channel_gate = nn.Sequential(nn.Linear(spec_channels, C // 8), nn.SiLU(inplace=True), nn.Linear(C // 8, C))
            nn.init.zeros_(self.spatial_gate.weight)
            nn.init.constant_(self.spatial_gate.bias, -2.0)  # start near identity: σ(-2)≈0.12
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.constant_(self.channel_gate[-1].bias, -2.0)
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(feat_dim, num_classes)
        self._last_attn = None

    def forward_features(self, x):
        Fm = self.backbone(x)[-1]
        if Fm.dim() == 4 and Fm.shape[1] != self.C and Fm.shape[-1] == self.C:  # NHWC archs
            Fm = Fm.permute(0, 3, 1, 2).contiguous()
        pooled = [Fm]
        if self.fusion == "none":
            return Fm.mean((-2, -1))
        sal = self.spec(x)
        G = self.spec_enc(sal)
        G = nn.functional.adaptive_avg_pool2d(G, Fm.shape[-2:])
        if self.fusion == "sga":
            a = torch.sigmoid(self.spatial_gate(torch.cat([self.reduce(Fm), G], 1)))
            s = torch.sigmoid(self.channel_gate(G.mean((-2, -1))))[:, :, None, None]
            Fm = Fm * (1 + a) * (1 + s)
            self._last_attn = a.detach()
        pooled = [Fm.mean((-2, -1)), G.mean((-2, -1))]
        return torch.cat(pooled, 1)

    def forward(self, x):
        f = self.forward_features(x)
        return self.head(self.drop(self.norm(f)))

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1, wd: float = 0.05):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        bb_ids = {id(p) for p in bb}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]
        return [{"params": bb, "lr": lr * backbone_lr_mult, "weight_decay": wd},
                {"params": rest, "lr": lr, "weight_decay": wd}]
