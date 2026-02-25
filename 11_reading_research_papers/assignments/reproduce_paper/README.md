# Reproduce a Paper

## Overview

Reproduce a published paper from scratch without copying the authors' code. Choose from: ResNet, Batch Normalization, DDPM, LoRA, or Vision Transformer. Implement the core method, run key experiments, compare results to the paper, and write a reproduction report.

## Learning Objectives

- Implement a paper's core method from description (not by copying code)
- Document every ambiguity and implementation decision
- Run controlled experiments with multiple seeds
- Compare results honestly against the paper's reported numbers
- Write a structured reproduction report

## Paper Options

1. **ResNet** (He et al., 2015) - Deep Residual Learning on CIFAR-10
2. **Batch Normalization** (Ioffe & Szegedy, 2015) - Training acceleration
3. **DDPM** (Ho et al., 2020) - Denoising Diffusion on MNIST/CIFAR-10
4. **LoRA** (Hu et al., 2021) - Low-rank adaptation of pretrained models
5. **Vision Transformer** (Dosovitskiy et al., 2020) - ViT on CIFAR-10

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train the paper's architecture
python train.py --config config.yaml

# Train with different seed
python train.py --config config.yaml --seed 1

# Evaluate and compare to paper
python evaluate.py --checkpoint checkpoints/best_model.pt

# Run ablation study
python evaluate.py --ablation
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Paper architecture stub (fill in based on chosen paper) |
| `data.py` | Dataset loading framework for the chosen paper's data |
| `train.py` | Training matching the paper's methodology |
| `evaluate.py` | Evaluation, results comparison, ablation study |
| `utils.py` | Experiment logging, result comparison helpers, decision log |
| `config.yaml` | Default hyperparameters (modify for chosen paper) |
| `notebooks/analysis.ipynb` | Results analysis and report generation |
