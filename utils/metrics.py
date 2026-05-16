"""MetricsTracker dataclass and accuracy utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MetricsTracker:
    """Track running averages of loss and top-1 accuracy over an epoch."""

    loss: float = field(default=0.0, init=False)
    acc: float = field(default=0.0, init=False)
    _loss_sum: float = field(default=0.0, init=False, repr=False)
    _acc_sum: float = field(default=0.0, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def update(self, loss: float, acc: float, n: int = 1) -> None:
        """Update running averages with a new batch result."""
        self._loss_sum += loss * n
        self._acc_sum += acc * n
        self._n += n
        self.loss = self._loss_sum / self._n
        self.acc = self._acc_sum / self._n

    def reset(self) -> None:
        self.loss = 0.0
        self.acc = 0.0
        self._loss_sum = 0.0
        self._acc_sum = 0.0
        self._n = 0

    def __repr__(self) -> str:
        return f"MetricsTracker(loss={self.loss:.4f}, acc={self.acc:.2f}%)"


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Compute top-k accuracy for each k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [
            correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size).item()
            for k in topk
        ]
