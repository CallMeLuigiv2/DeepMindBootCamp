"""
Diffusion Model - Utility Functions

Fully pre-written helpers for logging, metrics, visualization, and checkpointing.
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


def setup_logger(log_dir: str, name: str = "ddpm") -> logging.Logger:
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

    def plot(self, save_path: str, title: str = "Training Metrics") -> None:
        num_plots = len(self.metrics)
        if num_plots == 0:
            return
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
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


def save_checkpoint(path, model, optimizer, epoch, loss, extra=None):
    """Save a training checkpoint with optional extra data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        data["extra"] = extra
    torch.save(data, path)


def load_checkpoint(path, model, optimizer=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_sample_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    save_path: Optional[str] = None,
    title: str = "Generated Samples",
) -> None:
    """Display a grid of generated images.

    Args:
        samples: (N, C, H, W) tensor in [0, 1]
        nrow: Number of images per row
    """
    grid = make_grid(samples.clamp(0, 1), nrow=nrow, padding=2)
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


def plot_forward_process(
    x_0: torch.Tensor,
    alpha_bars: torch.Tensor,
    timesteps: List[int],
    save_path: Optional[str] = None,
) -> None:
    """Visualize the forward noising process for a single image.

    Shows the image at several increasing noise levels.

    Args:
        x_0: (1, C, H, W) - Clean image
        alpha_bars: (T,) - Noise schedule
        timesteps: List of timesteps to visualize
    """
    from model import forward_diffusion

    n = len(timesteps)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t])
        x_t, _ = forward_diffusion(x_0, t_tensor, alpha_bars)
        img = (x_t[0, 0].cpu() + 1) / 2  # Denormalize to [0, 1]
        axes[i].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"t={t}")
        axes[i].axis("off")

    fig.suptitle("Forward Diffusion Process", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_denoising_trajectory(
    trajectory: List[Tuple[int, torch.Tensor]],
    sample_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Visualize the reverse denoising process for a single sample.

    Args:
        trajectory: List of (timestep, x_t) tuples from sampling
        sample_idx: Which sample in the batch to display
    """
    n = len(trajectory)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    for i, (t, x_t) in enumerate(trajectory):
        img = (x_t[sample_idx, 0].cpu() + 1) / 2  # Denormalize
        axes[i].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"t={t}")
        axes[i].axis("off")

    fig.suptitle("Denoising Trajectory (reverse process)", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_noise_schedule_comparison(
    schedules: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
) -> None:
    """Plot alpha_bar_t curves for different noise schedules.

    Args:
        schedules: Dict mapping schedule name to alpha_bars tensor
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, alpha_bars in schedules.items():
        ax.plot(alpha_bars.numpy(), label=name, linewidth=2)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("alpha_bar_t (cumulative signal retained)")
    ax.set_title("Noise Schedule Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_noise_prediction_analysis(
    actual_noise: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: List[int],
    save_path: Optional[str] = None,
) -> None:
    """Visualize actual vs predicted noise at different timesteps.

    Shows actual noise, predicted noise, and residual for each timestep.
    """
    n = len(timesteps)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8))

    row_labels = ["Actual Noise", "Predicted Noise", "Residual"]
    for i, t in enumerate(timesteps):
        actual = actual_noise[i, 0].cpu().numpy()
        predicted = predicted_noise[i, 0].cpu().numpy()
        residual = actual - predicted

        axes[0, i].imshow(actual, cmap="RdBu", vmin=-2, vmax=2)
        axes[0, i].set_title(f"t={t}")
        axes[0, i].axis("off")

        axes[1, i].imshow(predicted, cmap="RdBu", vmin=-2, vmax=2)
        axes[1, i].axis("off")

        axes[2, i].imshow(residual, cmap="RdBu", vmin=-1, vmax=1)
        axes[2, i].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=30)

    fig.suptitle("Noise Prediction Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
