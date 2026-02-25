# Paper Improvement - Your First Research Contribution

## Overview

Take the paper you reproduced in Assignment 2, identify a genuine weakness, propose a specific improvement, implement it, test it with controlled A/B experiments, and write a research note. This is original research: the improvement might be small, might not work, but the process matters.

## Learning Objectives

- Identify non-obvious limitations in published methods through systematic analysis
- Formulate specific, testable hypotheses
- Design controlled experiments with proper baselines and ablations
- Report results honestly, including negative results
- Write a research note in workshop paper format

## Phases

1. **Identify a weakness** (5-8 hours): systematic analysis, select improvement, write hypothesis
2. **Implement the modification** (8-12 hours): localized change, keep original intact, A/B flag
3. **Run experiments** (8-12 hours): head-to-head comparison, ablation, diagnostic analysis
4. **Write the research note** (6-8 hours): ~4 pages in workshop paper format

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train baseline (original method from Assignment 2)
python train.py --config config.yaml --variant baseline

# Train with your improvement
python train.py --config config.yaml --variant improved

# Run A/B comparison across multiple seeds
python train.py --config config.yaml --ab-comparison

# Statistical comparison and ablation
python evaluate.py --compare --statistical
python evaluate.py --ablation
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Base model + improvement stubs (config flag to switch) |
| `data.py` | Dataset loading (same as reproduce_paper) |
| `train.py` | A/B experiment framework: baseline vs improvement |
| `evaluate.py` | Statistical comparison utilities, ablation visualization |
| `utils.py` | Ablation study helpers, hypothesis tracking, statistical tests |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Experiment analysis and research note drafting |
