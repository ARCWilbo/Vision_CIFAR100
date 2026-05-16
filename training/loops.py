"""Pure training and validation loop functions.

Both functions are stateless — they take all required objects as arguments
and return a MetricsTracker. Side effects (logging, checkpointing) belong
in the caller (training/train.py).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import MetricsTracker, topk_accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    aug_fn: Optional[object] = None,
) -> MetricsTracker:
    """Run one training epoch. Returns a MetricsTracker with averaged loss and top-1 acc."""
    model.train()
    metrics = MetricsTracker()
    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=device.type in ("cuda", "mps"))

    bar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)

        use_soft = aug_fn is not None
        if use_soft:
            images, soft_labels = aug_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(images)
            if use_soft:
                loss = -(soft_labels * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1 = topk_accuracy(logits.detach(), labels, topk=(1,))[0]
        metrics.update(loss.item(), acc1, n=images.size(0))
        bar.set_postfix(loss=f"{metrics.loss:.3f}", acc=f"{metrics.acc:.2f}%")

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> MetricsTracker:
    """Run one validation pass. Returns a MetricsTracker with averaged loss and top-1 acc."""
    model.eval()
    metrics = MetricsTracker()
    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=device.type in ("cuda", "mps"))

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)
        acc1 = topk_accuracy(logits, labels, topk=(1,))[0]
        metrics.update(loss.item(), acc1, n=images.size(0))

    return metrics
