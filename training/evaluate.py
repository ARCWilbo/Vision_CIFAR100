"""Test-set evaluation and prediction collection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import topk_accuracy

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset. Returns metrics and collected predictions."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=device.type in ("cuda", "mps"))

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            images, labels = images.to(device), labels.to(device)
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            n += bs
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    num_classes = int(max(labels_arr.max(), preds.max())) + 1
    conf_matrix = _confusion_matrix(preds, labels_arr, num_classes)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1).clip(min=1) * 100

    return {
        "loss": total_loss / n,
        "top1_acc": total_top1 / n,
        "top5_acc": total_top5 / n,
        "confusion_matrix": conf_matrix,
        "per_class_acc": per_class_acc,
        "class_names": CIFAR100_CLASSES[:num_classes],
        "predictions": preds,
        "labels": labels_arr,
    }


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        matrix[l, p] += 1
    return matrix


def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
) -> dict:
    """Measure inference latency and throughput."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times, total = [], 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            total += images.size(0)

    result = {
        "avg_batch_ms": float(np.mean(times)) * 1000,
        "throughput_samples_per_sec": total / sum(times),
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return result


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load model weights from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, {k: v for k, v in ckpt.items() if k != "model_state"}
