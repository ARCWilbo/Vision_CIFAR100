# CLAUDE.md

## Project Overview

This project trains and benchmarks deep learning image classification models on the CIFAR-100 dataset using PyTorch.

Primary goals:

- Build strong baseline CNN and ResNet models
- Experiment with augmentation and regularization
- Compare optimization techniques
- Analyze training dynamics and generalization
- Develop reproducible deep learning workflows

The project should prioritize:

- Clean modular code
- Reproducibility
- Experiment tracking
- Strong documentation
- Research-oriented structure

---

# Tech Stack

- Python 3.11+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm
- scikit-learn

Optional:

- wandb
- tensorboard
- albumentations

---

# Dataset

Dataset:

- CIFAR-100
- 100 image classes
- 32x32 RGB images

Dataset loading should use torchvision.datasets.CIFAR100.

Expected transforms:

- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- Normalize()

Dataset statistics:
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

---

# Initial Models

Implement:

1. SimpleCNN
2. ResNet18
3. ResNet34

All models must support:

- configurable num_classes
- configurable dropout
- device transfer support

---

# Training Features

Training pipeline must support:

- mixed precision training when available
- checkpoint saving
- early stopping
- gradient clipping
- learning rate scheduling
- validation metrics
- reproducible seeding

Metrics:

- train loss
- validation loss
- train accuracy
- validation accuracy

Save:

- best validation checkpoint
- final checkpoint
- training curves

---

# Augmentation Experiments

Implement support for:

- MixUp
- CutMix
- Label Smoothing
- Random Erasing

Experiments should be configurable from a single config file.

---

# Optimizers

Support:

- SGD + momentum
- AdamW

Schedulers:

- CosineAnnealingLR
- OneCycleLR

---

# CNN Project Structure

```text
CNN/
├── configs/
│   └── config.yaml                 # All hyperparameters
│
├── data/                           # CIFAR-100 dataset
│
├── models/
│   ├── cnn.py                      # Custom 3-block CNN (~700K params)
│   └── resnet.py                   # ResNet-18 with custom head
│
├── training/
│   ├── loops.py                    # train_one_epoch / validate (pure functions)
│   ├── evaluate.py                 # Test-set evaluation + prediction collection
│   └── train.py                    # Data loading, model/optimizer/scheduler builders, training loop
│
├── utils/
│   ├── seed.py                     # Deterministic seeding
│   ├── metrics.py                  # MetricsTracker dataclass
│   └── plots.py                    # Loss curves, accuracy curves, confusion  matrix, sample predictions
│
├── outputs/
│   ├── checkpoints/                # best_<arch>.pth, final_<arch>.pth
│   └── figures/                    # All saved plots
│
├── main.py                         # Entry point
│
└── requirements.txt
```

---

# Coding Standards

Requirements:

- Type hints everywhere
- Docstrings for all public functions
- Modular reusable functions
- Avoid duplicated logic
- Prefer composition over large monolithic scripts
- Use dataclasses for configs when appropriate

Code should be:

- research-friendly
- extensible
- readable
- reproducible

---

# Visualization Requirements

Generate:

- loss curves
- accuracy curves
- confusion matrices
- prediction visualizations
- class accuracy breakdowns

Use matplotlib only.

---

# Evaluation Goals

Target performance:

- SimpleCNN: 45-60%
- ResNet18: 70-78%
- ResNet34: 75-80%

Track:

- parameter count
- training time
- inference speed
- memory usage

---

# Reproducibility

All experiments must:

- set random seeds
- log hyperparameters
- save configs with checkpoints
- support deterministic behavior where possible

Seed:
42

---

# README Expectations

README should include:

- installation instructions
- training commands
- evaluation commands
- project structure
- sample results
- future improvements

---

# Future Extensions

Potential future work:

- Vision Transformers
- EfficientNet
- Self-supervised learning
- Contrastive learning
- Knowledge distillation
- CIFAR-100 hyperparameter sweeps
- Tiny ImageNet transfer

---

# Preferred Development Style

When implementing features:

1. Build minimal working version first
2. Verify correctness
3. Add modularity
4. Add optimization
5. Add experimentation support

Avoid premature optimization.

Favor correctness and clarity over cleverness.

---

# Training Defaults

Default hyperparameters:

- batch_size = 128
- epochs = 100
- optimizer = AdamW
- lr = 0.001
- weight_decay = 1e-4

Default device priority:

1. CUDA
2. MPS
3. CPU

---

# Deliverables

The final project should:

- train end-to-end with one command
- support configurable experiments
- generate publication-quality plots
- save reproducible checkpoints
- be structured like a small research repository
