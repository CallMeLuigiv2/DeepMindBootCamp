"""
VAE - Evaluation Script

Generates samples, visualizes latent space, and performs interpolation.

Usage:
    python evaluate.py --checkpoint checkpoints/best_vae_beta1.0.pt --visualize
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import get_mnist_dataloaders, flatten_batch, unflatten_batch
from model import VAE, ConditionalVAE
from utils import (
    set_seed,
    plot_sample_grid,
    plot_latent_space,
    plot_latent_grid_decode,
    plot_interpolation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


@torch.no_grad()
def generate_samples(model, num_samples, device, latent_dim=2):
    """Generate new images by sampling z ~ N(0, I) and decoding."""
    model.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    samples = model.decode(z)
    return unflatten_batch(samples)  # (num_samples, 1, 28, 28)


@torch.no_grad()
def encode_test_set(model, test_loader, device):
    """Encode the full test set and return (mu, labels)."""
    model.eval()
    all_mu = []
    all_labels = []

    for images, labels in test_loader:
        images = flatten_batch(images).to(device)
        mu, _ = model.encode(images)
        all_mu.append(mu.cpu())
        all_labels.append(labels)

    return torch.cat(all_mu), torch.cat(all_labels)


@torch.no_grad()
def reconstruct_samples(model, test_loader, device, num=16):
    """Reconstruct test images and return (originals, reconstructions)."""
    model.eval()
    images, labels = next(iter(test_loader))
    images_flat = flatten_batch(images[:num]).to(device)

    x_recon, _, _ = model(images_flat)

    return images[:num], unflatten_batch(x_recon.cpu())


@torch.no_grad()
def interpolate_latent(model, z_start, z_end, num_steps, device):
    """Linear interpolation between two latent points."""
    model.eval()
    alphas = torch.linspace(0, 1, num_steps)
    z_interp = torch.stack([(1 - a) * z_start + a * z_end for a in alphas])
    z_interp = z_interp.to(device)
    decoded = model.decode(z_interp)
    return unflatten_batch(decoded.cpu())


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    model_cfg = config["model"]
    vis_cfg = config["visualization"]

    # Load model
    # TODO: Reconstruct model architecture and load state_dict from checkpoint
    # model = VAE(...)
    # checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    _, test_loader = get_mnist_dataloaders(config["training"]["batch_size"])

    if args.visualize:
        # 1. Generate sample grid
        # samples = generate_samples(model, vis_cfg["num_samples"], device, model_cfg["latent_dim"])
        # plot_sample_grid(samples, save_path=os.path.join(args.output_dir, "samples.png"))

        # 2. Reconstructions
        # originals, recons = reconstruct_samples(model, test_loader, device)
        # # Display side by side

        # 3. Latent space visualization (only for latent_dim=2)
        # if model_cfg["latent_dim"] == 2:
        #     mu, labels = encode_test_set(model, test_loader, device)
        #     plot_latent_space(mu, labels, save_path=...)

        # 4. Decode grid of latent points
        # if model_cfg["latent_dim"] == 2:
        #     plot_latent_grid_decode(model, device, grid_size=20, range_=(-3, 3), save_path=...)

        # 5. Interpolation
        # plot_interpolation(...)

        print("TODO: Load model and generate visualizations")


if __name__ == "__main__":
    main()
