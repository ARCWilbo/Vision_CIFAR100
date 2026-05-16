#!/usr/bin/env python3
"""Entry point for training and evaluating CIFAR-100 classifiers.

Usage:
    # Train with default config
    python main.py

    # Train with config overrides
    python main.py --model resnet34 --epochs 50 --name my_run

    # Evaluate a saved checkpoint
    python main.py --eval --checkpoint outputs/checkpoints/default/best_resnet18.pth

    # Evaluate with plots and benchmark
    python main.py --eval --checkpoint outputs/checkpoints/default/best_resnet18.pth --plots --benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.seed import set_seed


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-100 classifier — train or evaluate")
    p.add_argument("--config",     type=str, default="configs/config.yaml", help="Path to config file")
    p.add_argument("--model",      type=str, choices = ["simplecnn", "resnet18", "resnet34"], required=True, help="Override model  (simplecnn|resnet18|resnet34)")
    # Evaluation flags
    p.add_argument("--eval",          action="store_true", help="Run evaluation instead of training")
    p.add_argument("--plots",         action="store_true", help="Save evaluation plots")
    p.add_argument("--benchmark",     action="store_true", help="Run inference benchmark")
    # Resume flags
    p.add_argument("--resume",        type=str, default=None, help="Path to .pth checkpoint to resume training from")
    p.add_argument("--resume-epochs", type=int, default=None, dest="resume_epochs", help="Number of additional epochs to train when resuming")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model:
        cfg["model"]["name"] = args.model
        cfg["name"] = args.model

    cfg["saved_checkpoint"] = str(Path(cfg["checkpoint"]["dir"]) / cfg["name"] / f"best_{cfg['model']['name']}.pth")

    if args.resume:
        cfg["resume_checkpoint"] = args.resume
        cfg["resume_epochs"]     = args.resume_epochs

    return cfg


def run_eval(cfg: dict, ckpt_path: str, device: torch.device, plots: bool, benchmark: bool) -> None:
    import numpy as np
    from training.train import build_dataloaders, build_model
    from training.evaluate import evaluate, load_checkpoint, benchmark_inference, CIFAR100_CLASSES
    from utils.plots import plot_confusion_matrix, plot_class_accuracy, plot_predictions

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    model, meta = load_checkpoint(model, ckpt_path, device)
    model.to(device)

    print(f"Checkpoint  : {ckpt_path}")
    print(f"Saved epoch : {meta.get('epoch', '?')}  |  Val acc: {meta.get('val_acc', '?'):.2f}%\n")

    results = evaluate(model, val_loader, device)
    print(f"Top-1 Accuracy : {results['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy : {results['top5_acc']:.2f}%")
    print(f"Val Loss       : {results['loss']:.4f}")

    acc = results["per_class_acc"]
    names = results["class_names"]
    worst = np.argsort(acc)[:5]
    best  = np.argsort(acc)[-5:][::-1]
    print("\nWorst 5:", "  ".join(f"{names[i]} {acc[i]:.0f}%" for i in worst))
    print("Best  5:", "  ".join(f"{names[i]} {acc[i]:.0f}%" for i in best))

    if benchmark:
        b = benchmark_inference(model, val_loader, device)
        print(f"\nLatency    : {b['avg_batch_ms']:.1f} ms/batch")
        print(f"Throughput : {b['throughput_samples_per_sec']:.0f} samples/s")
        if "peak_memory_mb" in b:
            print(f"Peak VRAM  : {b['peak_memory_mb']:.1f} MB")

    if plots:
        fig_dir = Path(cfg["outputs"]["figures"]) / cfg["name"]
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(results["confusion_matrix"], names,
                              save_path=fig_dir / "confusion_matrix.png",
                              title=f"{cfg['name']} — Confusion Matrix")
        plot_class_accuracy(acc, names,
                            save_path=fig_dir / "class_accuracy.png",
                            title=f"{cfg['name']} — Per-Class Accuracy")

        import torchvision
        from training.train import _MEAN, _STD
        import torchvision.transforms as T
        raw = torchvision.datasets.CIFAR100(cfg["data"]["root"], train=False, download=False,
                                            transform=T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)]))
        imgs = np.stack([raw[i][0].numpy() for i in range(32)])
        labs = np.array([raw[i][1] for i in range(32)])
        plot_predictions(imgs, labs, results["predictions"][:32], names,
                         save_path=fig_dir / "predictions.png")
        print(f"\nPlots saved to {fig_dir}/")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    cfg = apply_overrides(cfg, args)
    device = get_device()
    set_seed(cfg["training"]["seed"])

    if args.eval:
        if not cfg['saved_checkpoint']:
            print("Error: --checkpoint required for --eval mode.")
            sys.exit(1)
        run_eval(cfg, cfg['saved_checkpoint'], device, plots=args.plots, benchmark=args.benchmark)
    else:
        from training.train import run_training
        resume_path   = cfg.get("resume_checkpoint")
        resume_epochs = cfg.get("resume_epochs")
        if resume_path and resume_epochs is None:
            print("Error: --resume requires --resume-epochs N")
            sys.exit(1)
        print(f"Experiment : {cfg['name']}")
        print(f"Device     : {device}\n")
        run_training(cfg, device, resume_path=resume_path, resume_epochs=resume_epochs)


if __name__ == "__main__":
    main()
