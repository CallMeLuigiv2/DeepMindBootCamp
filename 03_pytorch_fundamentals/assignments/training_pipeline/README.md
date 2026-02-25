# Build a Complete Training Pipeline from Scratch

**Module 03 -- PyTorch Fundamentals, Sessions 3-6**

## Overview

Build a complete, production-quality training pipeline for a CNN on CIFAR-10. Every component -- data loading, model architecture, training loop, checkpointing, logging, early stopping -- must work together seamlessly.

**Target:** Achieve at least **85% test accuracy** on CIFAR-10 with a small CNN ($< 500K$ parameters).

### Learning Objectives

- Design a parameter-efficient CNN architecture
- Implement proper data augmentation and normalization
- Build a training loop with gradient clipping, LR scheduling, and early stopping
- Use TensorBoard for experiment logging
- Save and load checkpoints for reproducibility
- Write clean, modular, device-agnostic code

## Installation

From the project root:

```bash
pip install -e .
```

Then install module-specific dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Train with default config
python train.py --config config.yaml

# Train with custom hyperparameters
python train.py --epochs 100 --batch-size 128 --lr 0.01 --optimizer adamw --scheduler cosine

# Evaluate a saved checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pth

# View TensorBoard logs
tensorboard --logdir runs/
```

## Project Structure

| File | Description | What to Implement |
|------|-------------|-------------------|
| `model.py` | CNN architecture | `CIFAR10Net.__init__` and `forward` |
| `data.py` | Data loading pipeline | Already pre-written using `shared_utils` |
| `train.py` | Training script | `train_one_epoch`, `validate` loop bodies |
| `evaluate.py` | Standalone evaluation | Metric computation, per-class analysis |
| `utils.py` | Pre-written helpers | Nothing -- fully provided |
| `config.yaml` | Default hyperparameters | Adjust as needed |
| `notebooks/analysis.ipynb` | Visualization notebook | Training curves, experiment comparison |

## Architecture Constraints

- Total parameters must be **under 500K**
- At least 3 convolutional layers with increasing channel counts
- Must use `BatchNorm2d`, `MaxPool2d` or strided convolutions, and `Dropout`
- A fully connected classifier head

## Accuracy Target

**Minimum: 85% test accuracy.** Achievable with:

- A well-designed small CNN (~300K--500K parameters)
- Proper data augmentation (`RandomCrop` + `HorizontalFlip` at minimum)
- AdamW or SGD with momentum
- Cosine annealing or OneCycleLR
- 80--100 epochs of training

## Evaluation Criteria

| Component | Weight |
|-----------|--------|
| Correctness (runs, 85%+ accuracy, reproducible) | 40% |
| Code quality (modular, clean, PEP 8) | 30% |
| Engineering practices (argparse, logging, checkpoints) | 20% |
| Understanding (documented architecture choices) | 10% |
