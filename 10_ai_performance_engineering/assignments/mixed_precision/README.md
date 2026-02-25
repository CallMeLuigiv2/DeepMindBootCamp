# Mixed Precision Training and Quantization

## Overview

Implement mixed precision training (FP16/BF16) and post-training quantization (dynamic INT8, static INT8). Measure speed, memory, accuracy, and model size across all configurations. Produce a comprehensive comparison table and Pareto frontier analysis.

## Learning Objectives

- Implement mixed precision training with torch.amp.autocast and GradScaler
- Understand FP32 vs FP16 vs BF16 numerical properties
- Apply dynamic and static post-training quantization
- Create comprehensive performance comparison tables
- Analyze accuracy-speed tradeoffs (Pareto frontier)

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train FP32 baseline
python train.py --config config.yaml --precision fp32

# Train with FP16 mixed precision
python train.py --config config.yaml --precision fp16

# Train with BF16 mixed precision
python train.py --config config.yaml --precision bf16

# Run quantization and benchmarks
python evaluate.py --quantize --benchmark

# Generate comparison table and Pareto plot
python evaluate.py --compare-all
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | ResNet-18 with standard layers, compatible with quantization |
| `data.py` | CIFAR-10 loading for training and calibration |
| `train.py` | Training with configurable precision (FP32, FP16, BF16) |
| `evaluate.py` | Quantization, benchmarking, comparison tables, Pareto frontier |
| `utils.py` | Memory tracking, timing utilities, numerical precision exploration |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Numerical exploration and results visualization |
