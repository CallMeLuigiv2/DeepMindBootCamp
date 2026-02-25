# Profiling and Bottleneck Analysis

## Overview

Take a deliberately inefficient training pipeline, profile it systematically, identify every bottleneck, fix each one individually while measuring the improvement, and produce a professional profiling report. Target: at least 3x throughput improvement over baseline.

## Learning Objectives

- Profile GPU training pipelines with torch.profiler and Chrome traces
- Identify bottlenecks: data loading, CPU-GPU sync, batch size, augmentation
- Measure improvements incrementally (one fix at a time)
- Produce professional performance engineering reports

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Run deliberately slow baseline
python train.py --config config.yaml --mode baseline

# Profile the baseline
python train.py --config config.yaml --mode profile

# Run optimized pipeline
python train.py --config config.yaml --mode optimized

# Generate profiling report
python evaluate.py --report
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | ResNet-18 model variants (efficient vs inefficient) |
| `data.py` | Data loading with configurable anti-patterns (num_workers, pin_memory, augmentation) |
| `train.py` | Training with torch.profiler integration, baseline and optimized modes |
| `evaluate.py` | Profile comparison, visualization, waterfall chart generation |
| `utils.py` | Profiler context managers, timeline export, bottleneck identification helpers |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Interactive profiling analysis |
