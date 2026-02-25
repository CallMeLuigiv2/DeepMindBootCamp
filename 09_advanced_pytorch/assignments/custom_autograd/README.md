# Custom Autograd Mastery

## Overview

Implement production-quality custom autograd functions in PyTorch. You will build custom backward passes for a parameterized activation function, straight-through estimators for non-differentiable operations, and an asymmetric loss function. Each implementation is verified with `torch.autograd.gradcheck` and benchmarked against built-in alternatives.

## Learning Objectives

- Implement `torch.autograd.Function` subclasses with correct `forward` and `backward` methods
- Derive analytical gradients and verify them with `gradcheck`
- Use `ctx.save_for_backward` correctly for memory-efficient backward passes
- Implement straight-through estimators for non-differentiable operations
- Benchmark custom vs built-in implementations (forward/backward time, memory)

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train binary activation network with STE on MNIST
python train.py --config config.yaml

# Run gradient checks and correctness verification
python evaluate.py --verify-all

# Run performance benchmarks
python evaluate.py --benchmark

# Compare STE variants and asymmetric loss
python evaluate.py --compare-ste
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Custom autograd Functions (ParameterizedSwish, STE variants, AsymmetricMSE) and nn.Module wrappers |
| `data.py` | MNIST loading for binary activation network, synthetic data for regression |
| `train.py` | Training with custom autograd: STE binary network on MNIST, asymmetric loss regression |
| `evaluate.py` | Gradient verification, correctness tests, performance benchmarks |
| `utils.py` | Gradient checking utilities, timing helpers, memory measurement |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Interactive exploration of custom autograd functions |
