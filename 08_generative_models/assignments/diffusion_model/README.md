# Diffusion Model (DDPM)

## Overview

Build a Denoising Diffusion Probabilistic Model (DDPM) from scratch: the forward noising process, a U-Net denoising network with time conditioning, the training loop, and the iterative sampling procedure. Train on MNIST, visualize the denoising process, and experiment with noise schedules and diffusion steps.

## Learning Objectives

- Implement the forward diffusion process: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$
- Build a U-Net with sinusoidal time embeddings and skip connections
- Train with the simple noise prediction objective (MSE loss)
- Implement the DDPM reverse sampling loop
- Compare linear vs cosine noise schedules
- Understand how diffusion models generate structure progressively

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train the diffusion model
python train.py --config config.yaml

# Generate samples
python evaluate.py --checkpoint checkpoints/best_model.pt --generate --num-samples 64

# Visualize denoising trajectories
python evaluate.py --checkpoint checkpoints/best_model.pt --trajectories
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | U-Net with time embedding, noise schedule functions |
| `data.py` | MNIST/Fashion-MNIST loading with appropriate transforms |
| `train.py` | DDPM training loop with forward diffusion and noise prediction |
| `evaluate.py` | DDPM sampling loop, denoising trajectories, noise analysis |
| `utils.py` | Logging, metric tracking, visualization helpers |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Schedule comparison, T ablation, noise prediction analysis |
