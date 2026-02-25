# Hooks and Debugging Toolkit

## Overview

Build a comprehensive PyTorch debugging and instrumentation toolkit using hooks. You will implement feature extraction from pretrained models, gradient flow visualization, a gradient reversal layer for domain adaptation, an activation monitor for detecting pathologies, and a magnitude pruner using forward pre-hooks.

## Learning Objectives

- Register and manage forward hooks, backward hooks, and forward pre-hooks
- Extract intermediate features from pretrained models without modifying architecture
- Visualize gradient flow to diagnose vanishing/exploding gradients
- Implement gradient reversal for domain adaptation (DANN)
- Monitor activations to detect dead ReLUs and saturated sigmoids
- Apply magnitude pruning through hooks

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train with hooks enabled, capture activations and gradients
python train.py --config config.yaml

# Run gradient reversal domain adaptation experiment
python train.py --config config.yaml --experiment dann

# Visualize captured activations and gradient flow
python evaluate.py --visualize-gradients
python evaluate.py --visualize-activations

# Run pruning experiments
python evaluate.py --pruning-sweep
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Simple models for debugging: deep networks with pathological gradients, DANN architecture |
| `data.py` | MNIST loading with domain adaptation splits (noisy source, clean target) |
| `train.py` | Training with hooks enabled: gradient flow monitoring, DANN training, pruning experiments |
| `evaluate.py` | Visualization of activations, gradient flow plots, pruning accuracy curves |
| `utils.py` | Pre-written hook utilities: FeatureExtractor, GradientFlowVisualizer, ActivationMonitor, MagnitudePruner |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Interactive hook exploration and visualization |
