"""Data loading, model/optimizer/scheduler builders, and the main training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from models.resnet import ResNet18, ResNet34
from training.loops import train_one_epoch, validate
from training.augmentations import MixUp, CutMix
from utils.plots import plot_training_curves

_MEAN = (0.5071, 0.4867, 0.4408)
_STD = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-100."""
    acfg = cfg["augmentation"]
    pre_tensor: list = []
    post_tensor: list = []

    if acfg.get("random_crop", True):
        pre_tensor.append(T.RandomCrop(32, padding=4))
    if acfg.get("horizontal_flip", True):
        pre_tensor.append(T.RandomHorizontalFlip())
    if acfg.get("randaugment", False):
        pre_tensor.append(T.RandAugment(
            num_ops=acfg.get("randaugment_n", 4),
            magnitude=acfg.get("randaugment_m", 10),
        ))

    pre_tensor += [T.ToTensor(), T.Normalize(_MEAN, _STD)]

    if acfg.get("random_erasing", False):
        post_tensor.append(T.RandomErasing(p=acfg.get("random_erasing_prob", 0.5)))

    train_tf = T.Compose(pre_tensor + post_tensor)
    val_tf = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    data_root = cfg["data"]["root"]
    train_set = torchvision.datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tf)
    val_set   = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=val_tf)

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    pm = cfg["data"]["pin_memory"]

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pm, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=pm)
    return train_loader, val_loader


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from config."""
    name = cfg["model"]["name"].lower()
    nc = cfg["model"]["num_classes"]
    do = cfg["model"]["dropout"]
    registry = {
        "simplecnn": lambda: SimpleCNN(num_classes=nc, dropout=do),
        "resnet18":  lambda: ResNet18(num_classes=nc, dropout=do),
        "resnet34":  lambda: ResNet34(num_classes=nc, dropout=do),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]()


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tcfg = cfg["training"]
    if tcfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=tcfg["lr"],
            momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"], nesterov=True,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
    total_epochs: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    tcfg = cfg["training"]
    name = tcfg["scheduler"]
    epochs = total_epochs if total_epochs is not None else tcfg["epochs"]
    warmup = 0 if total_epochs is not None else tcfg["warmup_epochs"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1)
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tcfg["lr"] * 10,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            pct_start=warmup / epochs if epochs > 0 else 0.0,
        )
    return None


def build_augmentation(cfg: dict) -> Optional[object]:
    acfg = cfg["augmentation"]
    nc = cfg["model"]["num_classes"]

    candidates = []
    if acfg.get("cutmix_alpha", 0.0) > 0:
        candidates.append(CutMix(acfg["cutmix_alpha"], nc))
    if acfg.get("mixup_alpha", 0.0) > 0:
        candidates.append(MixUp(acfg["mixup_alpha"], nc))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Both enabled: exactly one chosen per batch at 50/50
    def _one_of(imgs, labs):
        return random.choice(candidates)(imgs, labs)
    return _one_of


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_training(
    cfg: dict,
    device: torch.device,
    resume_path: Optional[str | Path] = None,
    resume_epochs: Optional[int] = None,
) -> dict[str, list[float]]:
    """Full training run. Returns history dict with train/val loss and accuracy."""
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    aug_fn = build_augmentation(cfg)

    tcfg = cfg["training"]
    acfg = cfg["augmentation"]
    criterion = nn.CrossEntropyLoss(label_smoothing=acfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    early_stop = EarlyStopping(tcfg["early_stopping_patience"])

    out_dir = Path(cfg["checkpoint"]["dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    start_epoch  = 0
    best_val_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if resume_path is not None:
        from training.evaluate import load_checkpoint
        model, meta = load_checkpoint(model, resume_path, device)
        optimizer.load_state_dict(meta["optimizer_state"])
        start_epoch  = meta["epoch"]
        best_val_acc = meta.get("val_acc", 0.0)
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
        total_epochs = start_epoch + resume_epochs
        print(f"Resuming from epoch {start_epoch}  |  Best val acc so far: {best_val_acc:.2f}%\n")
    else:
        total_epochs = tcfg["epochs"]

    scheduler = build_scheduler(
        optimizer, cfg, len(train_loader),
        total_epochs=resume_epochs if resume_path is not None else None,
    )

    start = time.time()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}  |  Parameters: {n_params:,}  |  Device: {device}\n")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device,
                                  grad_clip=tcfg["grad_clip"], aug_fn=aug_fn)
        val_m   = validate(model, val_loader, criterion, device)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"loss {train_m.loss:.4f}/{val_m.loss:.4f} | "
            f"acc {train_m.acc:.2f}/{val_m.acc:.2f}% | "
            f"lr {lr:.2e} | {time.time()-start:.0f}s"
        )

        history["train_loss"].append(train_m.loss)
        history["val_loss"].append(val_m.loss)
        history["train_acc"].append(train_m.acc)
        history["val_acc"].append(val_m.acc)

        if val_m.acc > best_val_acc:
            best_val_acc = val_m.acc
            _save_checkpoint(model, optimizer, epoch, val_m.acc, cfg,
                             out_dir / f"best_{cfg['model']['name']}.pth")

        if early_stop.step(val_m.loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    # _save_checkpoint(model, optimizer, epoch, val_m.acc, cfg,
    #                  out_dir / f"final_{cfg['model']['name']}.pth")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig_path = Path(cfg["outputs"]["figures"]) / cfg["name"] / "training_curves.png"
    plot_training_curves(history, save_path=fig_path, title=f"{cfg['name']} — Training Curves")

    print(f"\nBest val acc: {best_val_acc:.2f}%")
    print(f"Training curves → {fig_path}")
    return history


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    cfg: dict,
    path: Path,
) -> None:
    torch.save({"epoch": epoch, "val_acc": val_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg}, path)
