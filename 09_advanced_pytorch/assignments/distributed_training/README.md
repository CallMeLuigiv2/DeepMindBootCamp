# Distributed Training and Advanced Techniques

## Overview

Take a single-GPU training pipeline and systematically upgrade it with DDP, gradient accumulation, mixed precision, gradient checkpointing, and torch.compile. Each technique is added incrementally with measurements, culminating in a production-quality combined training script.

## Learning Objectives

- Convert a single-GPU script to DistributedDataParallel (DDP)
- Implement gradient accumulation with correct loss normalization
- Add mixed precision training with autocast and GradScaler
- Apply gradient checkpointing for memory savings
- Use torch.compile for performance gains
- Combine all techniques into a single production-quality pipeline

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Single-GPU training (baseline)
python train.py --config config.yaml --mode single

# DDP training (multi-GPU or simulated)
torchrun --nproc_per_node=2 train.py --config config.yaml --mode ddp

# Full combined pipeline
torchrun --nproc_per_node=2 train.py --config config.yaml --mode full

# Evaluate and run ablation study
python evaluate.py --ablation
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | ResNet-18 compatible with DDP, gradient checkpointing, torch.compile |
| `data.py` | CIFAR-10 loading with DistributedSampler support |
| `train.py` | Training modes: single-GPU, DDP, full combined pipeline |
| `evaluate.py` | Ablation study: measure contribution of each technique |
| `utils.py` | DDP initialization helpers, gradient sync verification, metric aggregation |
| `config.yaml` | Default hyperparameters (num_gpus, backend, accumulation steps, etc.) |
| `notebooks/analysis.ipynb` | Ablation study visualization and analysis |
