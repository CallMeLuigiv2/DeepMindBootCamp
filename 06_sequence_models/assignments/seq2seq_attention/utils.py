"""
Seq2Seq with Attention - Utility Functions

Fully pre-written helpers for logging, metrics, visualization, and checkpointing.
"""

import logging
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


def setup_logger(log_dir: str, name: str = "seq2seq") -> logging.Logger:
    """Set up a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler
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
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> List[float]:
        """Get all recorded values for a metric."""
        return self.metrics.get(name, [])

    def get_last(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def plot(self, save_path: str, title: str = "Training Metrics") -> None:
        """Plot all tracked metrics and save to file."""
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(6 * len(self.metrics), 5))
        if len(self.metrics) == 1:
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


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
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
    """Load a training checkpoint.

    Returns:
        (epoch, best_val_loss) tuple
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


def visualize_attention(
    input_chars: List[str],
    output_chars: List[str],
    attention_weights: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Attention Weights",
) -> None:
    """Plot an attention heatmap.

    Args:
        input_chars: Characters in the source sequence
        output_chars: Characters in the generated output
        attention_weights: (output_len, input_len) numpy array
        save_path: Optional path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(max(8, len(input_chars) * 0.5), max(4, len(output_chars) * 0.4)))
    im = ax.imshow(attention_weights, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(input_chars)))
    ax.set_xticklabels(input_chars, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(len(output_chars)))
    ax.set_yticklabels(output_chars, fontsize=10)

    ax.set_xlabel("Input (source date)")
    ax.set_ylabel("Output (target date)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_accuracy_table(
    results: Dict[str, float], title: str = "Results"
) -> str:
    """Format accuracy results as a readable table string."""
    lines = [title, "=" * len(title)]
    max_key_len = max(len(k) for k in results)
    for key, value in results.items():
        lines.append(f"  {key:<{max_key_len}}  {value:.4f}  ({value*100:.1f}%)")
    return "\n".join(lines)
