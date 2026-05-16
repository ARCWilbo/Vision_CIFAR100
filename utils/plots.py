"""Loss curves, accuracy curves, confusion matrix, and sample predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: Optional[str | Path] = None,
    title: str = "Training Curves",
) -> None:
    """Plot loss and accuracy side-by-side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train")
    ax.plot(epochs, history["val_acc"], label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
    top_n: int = 20,
) -> None:
    """Heatmap of the top_n most-confused classes."""
    errors = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
    idx = np.argsort(errors)[-top_n:][::-1]
    sub = conf_matrix[np.ix_(idx, idx)].astype(float)
    norm = sub / sub.sum(axis=1, keepdims=True).clip(min=1)
    names = [class_names[i] for i in idx]
    n = len(idx)

    fig, ax = plt.subplots(figsize=(n * 0.55 + 1, n * 0.55 + 1))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = norm[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if val > thresh else "black")

    fig.tight_layout()
    _save(fig, save_path)


def plot_class_accuracy(
    per_class_acc: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str | Path] = None,
    title: str = "Per-Class Accuracy",
    top_n: int = 20,
) -> None:
    """Bar chart of the worst and best per-class accuracies."""
    sorted_idx = np.argsort(per_class_acc)
    half = top_n // 2
    idx = np.concatenate([sorted_idx[:half], sorted_idx[-half:]])
    accs = per_class_acc[idx]
    names = [class_names[i] for i in idx]
    colors = ["#d62728"] * half + ["#2ca02c"] * half

    fig, ax = plt.subplots(figsize=(10, max(4, len(idx) * 0.35)))
    ax.barh(range(len(idx)), accs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.axvline(float(per_class_acc.mean()), color="black", linestyle="--",
               linewidth=1, label=f"Mean {per_class_acc.mean():.1f}%")
    ax.legend(); ax.grid(True, axis="x", alpha=0.3); ax.set_xlim(0, 100)
    fig.tight_layout()
    _save(fig, save_path)


def plot_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str | Path] = None,
    n_rows: int = 4,
    n_cols: int = 8,
    mean: tuple[float, ...] = (0.5071, 0.4867, 0.4408),
    std: tuple[float, ...] = (0.2675, 0.2565, 0.2761),
) -> None:
    """Grid of sample images with true/predicted labels. Green = correct."""
    n = min(n_rows * n_cols, len(images))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.7))
    axes = np.array(axes).flatten()

    mean_a = np.array(mean).reshape(1, 1, 3)
    std_a = np.array(std).reshape(1, 1, 3)

    for i in range(n):
        ax = axes[i]
        img = np.clip(images[i].transpose(1, 2, 0) * std_a + mean_a, 0, 1)
        ax.imshow(img); ax.axis("off")
        color = "#2ca02c" if labels[i] == preds[i] else "#d62728"
        ax.set_title(
            f"T: {class_names[labels[i]]}\nP: {class_names[preds[i]]}",
            fontsize=5.5, color=color,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, save_path)
