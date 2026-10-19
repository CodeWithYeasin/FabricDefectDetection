"""Model registry.

Names
-----
``spectex_<timm_cnn>``     proposed SpecTexNet with the given CNN backbone, e.g. spectex_convnext_tiny
``<any timm model>``       fine-tuned baseline, e.g. resnet50, efficientnet_b0, convnext_tiny,
                           swin_tiny_patch4_window7_224, deit_small_patch16_224, mobilenetv3_large_100
"""
from __future__ import annotations

import timm
import torch.nn as nn

from .spectexnet import SpecTexNet


class TimmClassifier(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2,
                 drop_path_rate: float = 0.1):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        try:
            self.net = timm.create_model(name, drop_path_rate=drop_path_rate, **kw)
        except TypeError:
            self.net = timm.create_model(name, **kw)

    def forward(self, x):
        return self.net(x)

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1, wd: float = 0.05):
        head = self.net.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        bb = [p for p in self.net.parameters() if id(p) not in head_ids]
        return [{"params": bb, "lr": lr * backbone_lr_mult, "weight_decay": wd},
                {"params": list(head.parameters()), "lr": lr, "weight_decay": wd}]


def build_model(name: str, num_classes: int, pretrained: bool = True, **kw) -> nn.Module:
    if name.startswith("spectex_"):
        return SpecTexNet(backbone=name[len("spectex_"):], num_classes=num_classes, pretrained=pretrained, **kw)
    kw = {k: v for k, v in kw.items() if k in {"drop_rate", "drop_path_rate"}}
    return TimmClassifier(name, num_classes, pretrained=pretrained, **kw)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature layer, used by Grad-CAM."""
    if isinstance(model, SpecTexNet):
        return model.backbone
    net = model.net
    for attr in ("stages", "layer4", "blocks", "features", "layers"):
        if hasattr(net, attr):
            m = getattr(net, attr)
            return m[-1] if isinstance(m, (nn.Sequential, nn.ModuleList)) else m
    if hasattr(net, "conv_head"):
        return net.conv_head
    raise ValueError("cannot find a Grad-CAM layer for this model")
