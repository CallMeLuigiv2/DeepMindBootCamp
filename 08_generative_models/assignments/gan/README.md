# Generative Adversarial Network (GAN)

## Overview

Build GANs of increasing sophistication: basic MLP GAN on MNIST, DCGAN on CIFAR-10/CelebA, and WGAN-GP. Experience training instabilities (mode collapse, discriminator dominance), then implement architectural and objective improvements. Evaluate with FID scores.

## Learning Objectives

- Implement a basic GAN with alternating generator/discriminator training
- Experience and diagnose GAN failure modes: mode collapse, discriminator dominance
- Build a DCGAN following the architecture guidelines
- Implement Wasserstein distance loss with gradient penalty (WGAN-GP)
- Quantitatively evaluate generative quality with FID scores

## Installation

```bash
pip install -e ../../../
pip install -r requirements.txt
```

## How to Run

```bash
# Train basic MLP GAN on MNIST
python train.py --config config.yaml --model basic

# Train DCGAN on CIFAR-10
python train.py --config config.yaml --model dcgan --dataset cifar10

# Train WGAN-GP
python train.py --config config.yaml --model wgan_gp --dataset cifar10

# Evaluate and compute FID
python evaluate.py --checkpoint checkpoints/dcgan_best.pt --compute-fid
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Generator and Discriminator for basic GAN, DCGAN, and WGAN-GP |
| `data.py` | MNIST, CIFAR-10, CelebA loading with proper normalization |
| `train.py` | Alternating G/D training, label smoothing, WGAN critic loop |
| `evaluate.py` | FID computation, sample generation, training progression |
| `utils.py` | Logging, metric tracking, sample grid visualization, weight init |
| `config.yaml` | Default hyperparameters for all GAN variants |
| `notebooks/analysis.ipynb` | Instability experiments, DCGAN vs WGAN-GP comparison |
