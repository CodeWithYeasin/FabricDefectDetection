"""Loss functions for long-tailed defect data.

``cb_focal`` = class-balanced weights from the effective number of samples (Cui et al., CVPR 2019)
combined with focal modulation (Lin et al., ICCV 2017).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(counts, beta: float = 0.999) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-8)
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        ce = F.nll_loss(logp, target, reduction="none")
        if self.ls > 0:
            ce = (1 - self.ls) * ce - self.ls * logp.mean(-1)
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        loss = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight.to(loss.device)[target]
        return loss.mean()


def build_loss(name: str, class_counts, label_smoothing: float = 0.0, gamma: float = 2.0, beta: float = 0.999):
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "wce":
        return nn.CrossEntropyLoss(weight=class_balanced_weights(class_counts, beta), label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma, None, label_smoothing)
    if name == "cb_focal":
        return FocalLoss(gamma, class_balanced_weights(class_counts, beta), label_smoothing)
    raise ValueError(name)
