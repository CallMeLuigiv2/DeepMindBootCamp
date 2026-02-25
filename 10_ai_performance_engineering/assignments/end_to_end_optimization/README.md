# End-to-End Optimization - The "Make It Fast" Challenge

## Overview

Take a naive Transformer encoder training pipeline for text classification and apply every optimization technique: optimized DataLoader, mixed precision, torch.compile, gradient accumulation, gradient checkpointing, Flash Attention, and optimal batch sizing. Measure each individually and produce a professional performance report.

## Learning Objectives

- Build a Transformer encoder from specification and compute parameter/memory budgets
- Apply a full suite of optimization techniques incrementally
- Measure and attribute speedup from each technique
- Produce a professional performance engineering report with waterfall chart

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train naive baseline
python train.py --config config.yaml --mode baseline

# Train fully optimized
python train.py --config config.yaml --mode optimized

# Run benchmark suite (all configurations)
python evaluate.py --benchmark-suite

# Generate performance report
python evaluate.py --report
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Transformer encoder for text classification (from specification) |
| `data.py` | Optimized data pipeline: pre-tokenization, prefetching, pinned memory |
| `train.py` | Baseline and optimized training: all techniques combined |
| `evaluate.py` | Benchmark suite, waterfall chart, performance report |
| `utils.py` | Throughput measurement, memory budget calculator, benchmark harness |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Performance analysis and report generation |
