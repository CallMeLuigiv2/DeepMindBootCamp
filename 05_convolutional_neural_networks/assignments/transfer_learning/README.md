# Transfer Learning Project

**Module 05 -- Convolutional Neural Networks, Assignment 3**

## Overview

Take a pretrained model, adapt it to a domain-specific classification task, compare three transfer learning strategies rigorously, and build interpretability into your pipeline with Grad-CAM. This is the workflow used at production ML teams when given a new vision task.

### Learning Objectives

- Fine-tune pretrained ImageNet models for a new domain
- Compare three strategies: frozen backbone, partial fine-tuning, full fine-tuning
- Implement differential learning rates for progressive unfreezing
- Use two-phase training (warmup then fine-tune)
- Build Grad-CAM visualizations for model interpretability
- Conduct rigorous evaluation with confusion matrices and per-class analysis

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
# Train with default config (Strategy 1: Frozen backbone)
python train.py --config config.yaml --strategy frozen

# Train Strategy 2: Partial fine-tuning
python train.py --config config.yaml --strategy partial

# Train Strategy 3: Full fine-tuning with differential LR
python train.py --config config.yaml --strategy full

# Train from-scratch baseline
python train.py --config config.yaml --strategy scratch

# Train all strategies for comparison
python train.py --config config.yaml --compare-all

# Evaluate with Grad-CAM
python evaluate.py --checkpoint checkpoints/full_best.pth --gradcam
```

## Project Structure

| File | Description | What to Implement |
|------|-------------|-------------------|
| `model.py` | Model creation for each strategy | Freeze/unfreeze logic, classifier heads |
| `data.py` | Dataset loading with ImageFolder | Custom dataset transforms |
| `train.py` | Two-phase training script | Training loop bodies |
| `evaluate.py` | Evaluation + Grad-CAM | Grad-CAM generation, metric computation |
| `utils.py` | Pre-written helpers | Nothing -- fully provided |
| `config.yaml` | Strategy-specific hyperparameters | Adjust as needed |
| `notebooks/analysis.ipynb` | Comparison and Grad-CAM analysis | Plots, written report |

## Recommended Datasets

Choose ONE (or bring your own with $\geq 5$ classes and $\geq 100$ images/class):

1. **Oxford Flowers 102** -- `torchvision.datasets.Flowers102`
2. **Food-101** -- `torchvision.datasets.Food101` (use a 10--20 class subset)
3. **EuroSAT** -- `torchvision.datasets.EuroSAT` (satellite imagery)
4. **Stanford Cars** -- 196 fine-grained car classes
5. **Intel Image Classification** -- 6 scene classes (Kaggle)

## Transfer Learning Strategies

| Strategy | Trainable Layers | Learning Rate | Epochs |
|----------|-----------------|---------------|--------|
| Baseline (scratch) | All (small CNN) | $10^{-3}$ | 50 |
| Strategy 1 (frozen) | Head only | $10^{-3}$ | 30 |
| Strategy 2 (partial) | layer4 + head | $10^{-4}$ (layer4), $10^{-3}$ (head) | 30 |
| Strategy 3 (full) | All (differential LR) | $10^{-6}$ to $10^{-3}$ | 5 warmup + 25 fine-tune |

## Evaluation Criteria

| Component | Weight |
|-----------|--------|
| Task selection and data prep | 10% |
| Three transfer strategies | 25% |
| Data augmentation study | 10% |
| Grad-CAM implementation | 20% |
| Evaluation rigor | 15% |
| Report quality | 20% |
