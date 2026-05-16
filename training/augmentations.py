"""MixUp and CutMix augmentations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class MixUp:
    """Convex combination of two samples and their one-hot labels."""

    def __init__(self, alpha: float = 0.4, num_classes: int = 100) -> None:
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[idx]
        la = F.one_hot(labels, self.num_classes).float()
        lb = F.one_hot(labels[idx], self.num_classes).float()
        return mixed, lam * la + (1 - lam) * lb


class CutMix:
    """Paste a rectangular patch from one image onto another."""

    def __init__(self, alpha: float = 1.0, num_classes: int = 100) -> None:
        self.alpha = alpha
        self.num_classes = num_classes

    def _rand_bbox(self, W: int, H: int, lam: float) -> tuple[int, int, int, int]:
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        return (
            max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0),
            min(cx + cut_w // 2, W), min(cy + cut_h // 2, H),
        )

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        x1, y1, x2, y2 = self._rand_bbox(images.size(2), images.size(3), lam)
        mixed = images.clone()
        mixed[:, :, x1:x2, y1:y2] = images[idx, :, x1:x2, y1:y2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3))
        la = F.one_hot(labels, self.num_classes).float()
        lb = F.one_hot(labels[idx], self.num_classes).float()
        return mixed, lam * la + (1 - lam) * lb
