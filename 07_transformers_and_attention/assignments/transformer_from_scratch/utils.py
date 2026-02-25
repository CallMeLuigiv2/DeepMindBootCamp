"""
Transformer from Scratch - Utility Functions

Fully pre-written helpers for logging, LR scheduling, metrics, and visualization.
"""

import logging
import math
import os
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, name: str = "transformer") -> logging.Logger:
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


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, self.metrics.items()):
            ax.plot(range(1, len(values) + 1), values, linewidth=1)
            ax.set_xlabel("Step / Epoch")
            ax.set_ylabel(name)
            ax.set_title(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with linear warmup.

    If total_steps is provided, uses linear warmup then linear decay.
    Otherwise, uses the Transformer schedule: lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (for linear decay schedule)

    Returns:
        LambdaLR scheduler
    """
    if total_steps is not None:
        def lr_lambda(step):
            step = max(step, 1)
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)
    else:
        def lr_lambda(step):
            step = max(step, 1)
            return min(step ** (-0.5), step * warmup_steps ** (-1.5)) * warmup_steps ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_or_epoch: int,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step_or_epoch": step_or_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step_or_epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def plot_attention_heads(
    attention_weights: torch.Tensor,
    tokens: List[str],
    layer: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Plot attention heatmaps for all heads in a given layer.

    Args:
        attention_weights: (num_heads, seq_len, seq_len)
        tokens: List of token strings for axis labels
        layer: Layer index (for the title)
        save_path: Optional path to save the figure
    """
    num_heads = attention_weights.size(0)
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attention_weights[h].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        if len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=7)

    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Layer {layer} Attention Heads", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_positional_encoding(pe_matrix: np.ndarray, save_path: Optional[str] = None) -> None:
    """Visualize the positional encoding matrix as a heatmap.

    Args:
        pe_matrix: (max_len, d_model) numpy array
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pe_matrix, cmap="RdBu", aspect="auto")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Position")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
