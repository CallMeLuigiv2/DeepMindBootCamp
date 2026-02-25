"""
VAE - Utility Functions

Fully pre-written helpers for logging, metrics, and visualization.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, name: str = "vae") -> logging.Logger:
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
    """Track and plot training metrics over epochs."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> List[float]:
        return self.metrics.get(name, [])

    def plot(self, save_path: str, title: str = "Training Metrics") -> None:
        num_plots = len(self.metrics)
        if num_plots == 0:
            return
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        for ax, (name, values) in zip(axes, self.metrics.items()):
            ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def plot_sample_grid(
    samples: torch.Tensor,
    nrow: int = 10,
    save_path: Optional[str] = None,
    title: str = "Generated Samples",
) -> None:
    """Display a grid of generated images.

    Args:
        samples: (N, 1, 28, 28) tensor of images
        nrow: Number of images per row
        save_path: Optional path to save
        title: Plot title
    """
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """Plot original and reconstructed images side by side."""
    n = originals.size(0)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))

    for i in range(n):
        axes[0, i].imshow(originals[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstructed", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latent_space(
    mu: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Latent Space (2D)",
) -> None:
    """Scatter plot of encoded test set colored by digit class.

    Args:
        mu: (N, 2) - Encoded means
        labels: (N,) - Digit labels
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        mu[:, 0].numpy(), mu[:, 1].numpy(),
        c=labels.numpy(), cmap="tab10", s=1, alpha=0.5,
    )
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, ticks=range(10), label="Digit Class")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


@torch.no_grad()
def plot_latent_grid_decode(
    model,
    device: torch.device,
    grid_size: int = 20,
    range_: Tuple[float, float] = (-3.0, 3.0),
    save_path: Optional[str] = None,
) -> None:
    """Decode a grid of latent points and display as image grid."""
    model.eval()
    z1 = np.linspace(range_[0], range_[1], grid_size)
    z2 = np.linspace(range_[1], range_[0], grid_size)  # Flip for image y-axis

    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    for i, z2_val in enumerate(z2):
        for j, z1_val in enumerate(z1):
            z = torch.tensor([[z1_val, z2_val]], dtype=torch.float32, device=device)
            x_decoded = model.decode(z).cpu().numpy().reshape(28, 28)
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = x_decoded

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray")
    ax.set_title("Decoded Latent Space Grid")
    ax.set_xlabel(f"z_1 ({range_[0]} to {range_[1]})")
    ax.set_ylabel(f"z_2 ({range_[0]} to {range_[1]})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_interpolation(
    images: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Latent Space Interpolation",
) -> None:
    """Display a sequence of interpolated images in a single row."""
    n = images.size(0)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    for i in range(n):
        axes[i].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        axes[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
