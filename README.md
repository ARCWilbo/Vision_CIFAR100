# CIFAR-100 Deep Learning Benchmark

Train and benchmark CNN / ResNet classifiers on CIFAR-100 with a reproducible, research-oriented pipeline.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ and PyTorch 2.0+ are required.

---

## Quick Start

### Train a model

```bash
python main.py --model resnet18
python main.py --model resnet34
python main.py --model simplecnn
```

### Resume training from a checkpoint

```bash
# Continue training a saved checkpoint for N additional epochs
python main.py --model resnet18 \
  --resume outputs/checkpoints/resnet18/best_resnet18.pth \
  --resume-epochs 20
```

---

## Evaluation

```bash
# Basic evaluation
python main.py --eval --checkpoint outputs/checkpoints/default/best_resnet18.pth

# With plots (confusion matrix, class accuracy, prediction grid)
python main.py --eval --checkpoint outputs/checkpoints/default/best_resnet34.pth --plots

# With inference benchmark (latency + throughput)
python main.py --eval --checkpoint outputs/checkpoints/default/best_resnet34.pth --plots --benchmark
```

---

## Project Structure

```
CIFAR100/
├── configs/
│   └── config.yaml                 # All hyperparameters
│
├── data/                           # CIFAR-100 dataset (auto-downloaded)
│
├── models/
│   ├── cnn.py                      # Custom 3-block CNN (~838K params)
│   └── resnet.py                   # ResNet-18/34 with CIFAR-adapted stem
│
├── training/
│   ├── loops.py                    # train_one_epoch / validate (pure functions)
│   ├── evaluate.py                 # Test-set evaluation + prediction collection
│   ├── train.py                    # Data loading, model/optimizer/scheduler builders, training loop
│   └── augmentations.py            # MixUp, CutMix
│
├── utils/
│   ├── seed.py                     # Deterministic seeding
│   ├── metrics.py                  # MetricsTracker dataclass
│   └── plots.py                    # Loss curves, accuracy curves, confusion matrix, sample predictions
│
├── outputs/
│   ├── checkpoints/                # best_<arch>.pth, final_<arch>.pth
│   └── figures/                    # All saved plots
│
├── main.py                         # Entry point (train + eval)
└── requirements.txt
```

---

## Configuration

All experiments are driven by `configs/config.yaml`. Any field can be overridden with a CLI flag.

```yaml
name: default

model:
  name: resnet18 # simplecnn | resnet18 | resnet34
  num_classes: 100
  dropout: 0.1

training:
  epochs: 100
  batch_size: 128
  optimizer: adamw # sgd | adamw
  lr: 0.001
  scheduler: cosine # cosine | onecycle
  early_stopping_patience: 20
  seed: 42

augmentation:
  mixup_alpha: 0.0
  cutmix_alpha: 1.0 # enable CutMix
  label_smoothing: 0.1
  random_erasing: true

outputs:
  figures: ./outputs/figures
```

---

## Models

| Model     | Parameters | Top-1 Target | Top-1 Achieved |
| --------- | ---------- | ------------ | -------------- |
| SimpleCNN | ~838K      | 45–60%       | —              |
| ResNet18  | ~11.2M     | 70–78%       | **72.86%**     |
| ResNet34  | ~21.3M     | 75–80%       | 68.81%         |

All models use a **CIFAR-adapted stem** (3×3 conv, stride 1, no MaxPool) to preserve spatial resolution on 32×32 inputs.

---

## Training Features

- Mixed-precision training (AMP) on CUDA and MPS
- Gradient clipping
- Cosine annealing or OneCycleLR scheduling with linear warmup
- Early stopping on validation loss
- Best and final checkpoint saving (`best_<arch>.pth`, `final_<arch>.pth`)
- Config saved alongside each checkpoint for reproducibility

## Augmentations

- Standard: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- Optional: **MixUp**, **CutMix**, **Label Smoothing**, **RandomErasing**

---

## Results — ResNet18

Trained with AdamW (lr=1e-3, cosine LR, label smoothing=0.1, seed=42). 35 epochs trained.

| Metric         | Value  |
| -------------- | ------ |
| Top-1 Accuracy | 72.86% |
| Val Loss       | 1.6947 |
| Epochs trained | 35     |
| Parameters     | 11.2M  |
| Optimizer      | AdamW  |
| Scheduler      | Cosine |

> ResNet18 comfortably hits the 70–78% target at 72.86%. It also outperforms the larger ResNet34 run — the deeper model stopped early at epoch 39 before fully converging, suggesting it would benefit from a longer warmup or extended training (which `--resume` can now handle).

---

## Results — ResNet34

Trained with AdamW (lr=1e-3, cosine LR, label smoothing=0.1, seed=42). Early stopping triggered at epoch 39.

| Metric         | Value  |
| -------------- | ------ |
| Top-1 Accuracy | 68.81% |
| Top-5 Accuracy | 88.41% |
| Val Loss       | 1.3342 |
| Epochs trained | 39     |
| Parameters     | 21.3M  |
| Optimizer      | AdamW  |
| Scheduler      | Cosine |

**Best 5 classes** (per-class accuracy):

| Class      | Accuracy |
| ---------- | -------- |
| Sunflower  | 94.0%    |
| Motorcycle | 94.0%    |
| Orange     | 94.0%    |
| Road       | 94.0%    |
| Plain      | 92.0%    |

**Worst 5 classes** (per-class accuracy):

| Class     | Accuracy |
| --------- | -------- |
| Lizard    | 35.0%    |
| Seal      | 36.0%    |
| Boy       | 36.0%    |
| Butterfly | 39.0%    |
| Otter     | 41.0%    |

> **Note:** This ResNet34 run used a weaker augmentation configuration than the ResNet18 run (no RandAugment, weaker CutMix/MixUp settings), which led to overfitting and explains why the larger model underperforms the smaller one. ResNet34 needs to be retrained with the full augmentation pipeline to reach its target range. Retraining is pending — training time on a local Mac (MPS) is prohibitively slow for a 100-epoch run at this scale.

---

## Future Extensions

- Vision Transformers (ViT-S/16)
- EfficientNet-B0
- Self-supervised pretraining (SimCLR / BYOL)
- Knowledge distillation from ResNet34 → SimpleCNN
- Hyperparameter sweeps (Optuna / W&B)
- Tiny-ImageNet transfer learning
