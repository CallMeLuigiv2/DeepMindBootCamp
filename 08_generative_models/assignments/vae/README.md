# Variational Autoencoder (VAE)

## Overview

Build a Variational Autoencoder from scratch: the encoder that outputs distribution parameters, the reparameterization trick, the decoder, and the ELBO loss. Train it on MNIST to generate handwritten digits, explore the latent space, and experiment with the beta-VAE tradeoff and conditional generation.

## Learning Objectives

- Implement the VAE architecture: encoder -> (mu, logvar) -> reparameterize -> decoder
- Understand the ELBO: $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$
- Visualize and navigate the 2D latent space
- Explore the reconstruction-regularization tradeoff (beta-VAE)
- Build a Conditional VAE for class-specific generation

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train the standard VAE
python train.py --config config.yaml

# Train beta-VAE with different beta values
python train.py --config config.yaml --beta 0.5
python train.py --config config.yaml --beta 5.0

# Train Conditional VAE
python train.py --config config.yaml --conditional

# Generate samples and visualize latent space
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | VAE (encoder, reparameterize, decoder), CVAE, loss function |
| `data.py` | MNIST loading, normalization, DataLoader creation |
| `train.py` | Training loop with ELBO loss, KL annealing, checkpointing |
| `evaluate.py` | Sample generation, latent space visualization, interpolation |
| `utils.py` | Logging, metric tracking, grid/interpolation visualization |
| `config.yaml` | Default hyperparameters |
| `notebooks/analysis.ipynb` | Latent space exploration, beta-VAE comparison |
