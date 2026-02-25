"""
GAN - Utility Functions

Fully pre-written helpers for logging, weight initialization, sample visualization,
and metric tracking.
"""

import logging
import os
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, name: str = "gan") -> logging.Logger:
    """Set up a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    return logger


class MetricTracker:
    """Track and plot training metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> List[float]:
        return self.metrics.get(name, [])

    def plot(self, save_path: str, title: str = "GAN Training") -> None:
        num_plots = len(self.metrics)
        if num_plots == 0:
            return
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        for ax, (name, values) in zip(axes, self.metrics.items()):
            ax.plot(range(1, len(values) + 1), values, linewidth=1)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_checkpoint(path, model_g, model_d, opt_g, opt_d, epoch):
    """Save GAN training checkpoint (both networks)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "generator_state_dict": model_g.state_dict(),
        "discriminator_state_dict": model_d.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
    }, path)


def weights_init(m: nn.Module) -> None:
    """DCGAN weight initialization.

    Conv layers: normal(0, 0.02)
    BatchNorm layers: normal(1, 0.02), bias=0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def save_sample_grid(
    generator: nn.Module,
    latent_dim: int,
    device: torch.device,
    num_samples: int = 64,
    save_path: str = "samples.png",
    nrow: int = 8,
) -> None:
    """Generate a grid of fake images and save to file."""
    generator.eval()

    # Handle different generator input shapes
    z = torch.randn(num_samples, latent_dim, device=device)

    # Check if generator expects 4D input (DCGAN)
    try:
        fake = generator(z.view(num_samples, latent_dim, 1, 1))
    except RuntimeError:
        fake = generator(z)

    # Denormalize from [-1, 1] to [0, 1]
    fake = (fake + 1) / 2

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(fake, save_path, nrow=nrow, padding=2)
    generator.train()


def plot_training_progression(
    sample_paths: List[str],
    epoch_labels: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Display sample grids from multiple epochs side by side."""
    n = len(sample_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, path, label in zip(axes, sample_paths, epoch_labels):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(f"Epoch {label}")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_d_scores(
    d_real_scores: List[float],
    d_fake_scores: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot D(x_real) and D(G(z)) over training to diagnose training dynamics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d_real_scores, label="D(x_real)", linewidth=1)
    ax.plot(d_fake_scores, label="D(G(z))", linewidth=1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equilibrium")
    ax.set_xlabel("Step")
    ax.set_ylabel("Discriminator Output")
    ax.set_title("Discriminator Scores Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
