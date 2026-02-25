# Build the Landmark CNN Architectures

**Module 05 -- Convolutional Neural Networks, Assignment 2**

## Overview

Implement three historically significant CNN architectures from scratch -- LeNet-5, VGG-11, and ResNet-18 -- train them on CIFAR-10, and conduct a rigorous head-to-head comparison. Then run the critical skip connection experiment to understand why residual connections transformed deep learning.

### Learning Objectives

- Implement LeNet-5, VGG-11, and ResNet-18 as PyTorch `nn.Module` classes
- Count parameters manually and verify against code
- Train all architectures with identical settings for fair comparison
- Analyze the impact of skip connections via PlainNet-18 vs ResNet-18
- Understand depth vs width tradeoffs and gradient flow dynamics

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
# Train all architectures with default config
python train.py --config config.yaml

# Train a specific architecture
python train.py --config config.yaml --arch resnet18

# Train the PlainNet variant (no skip connections)
python train.py --config config.yaml --arch plainnet18

# Evaluate a saved checkpoint
python evaluate.py --checkpoint checkpoints/resnet18_best.pth --arch resnet18

# Run the full comparison
python train.py --config config.yaml --compare-all
```

## Project Structure

| File | Description | What to Implement |
|------|-------------|-------------------|
| `model.py` | Architecture definitions | LeNet5, VGG11, BasicBlock, ResNet18 |
| `data.py` | CIFAR-10 data loading | Already pre-written |
| `train.py` | Training with architecture selection | Training/validation loop bodies |
| `evaluate.py` | Evaluation and comparison | Metric computation, comparison plots |
| `utils.py` | Pre-written helpers | Nothing -- fully provided |
| `config.yaml` | Per-architecture hyperparameters | Adjust as needed |
| `notebooks/analysis.ipynb` | Head-to-head analysis notebook | Comparison plots, written analysis |

## Architecture Summary

| Model | Parameters | Expected CIFAR-10 Accuracy |
|-------|-----------|---------------------------|
| LeNet-5 | ~62K | $>75\%$ |
| VGG-11 (GAP) | ~9.2M | $>88\%$ |
| ResNet-18 | ~11.2M | $>92\%$ |
| PlainNet-18 | ~11.2M | Lower than ResNet-18 |

## Key Experiments

### Head-to-Head Comparison (Part 4)
Train all three architectures on CIFAR-10 with identical training setup:
- Data augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
- LR schedule: divide by 10 at epochs 80 and 120
- 150 epochs, batch size 128

### Skip Connection Experiment (Part 5)
Remove skip connections from ResNet-18 to create PlainNet-18.
Compare training dynamics, final accuracy, and gradient norms.

## Evaluation Criteria

| Component | Weight |
|-----------|--------|
| Architecture correctness (shapes, params) | 25% |
| Training pipeline | 15% |
| Training results (reasonable accuracies) | 15% |
| Comparison analysis (table, plots, writing) | 20% |
| Skip connection experiment + gradient analysis | 15% |
| Code quality | 10% |
