"""
Fine-Tune Pretrained Transformer - Utility Functions

Fully pre-written helpers for logging, metrics, visualization, and checkpointing.
"""

import logging
import os
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, name: str = "finetune") -> logging.Logger:
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


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute classification metrics.

    Returns:
        Dictionary with 'accuracy', 'f1', and 'confusion_matrix'.
    """
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
    }


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


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
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
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer: int = 0,
    head: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Plot attention weights as a heatmap for a single head."""
    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(6, len(tokens) * 0.4)))
    im = ax.imshow(attention_weights, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_title(f"Layer {layer}, Head {head}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_data_efficiency(
    data_sizes: List[int],
    bert_accuracies: List[float],
    scratch_accuracies: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy vs training data size for both approaches."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data_sizes, bert_accuracies, "o-", label="Fine-tuned BERT", linewidth=2)
    ax.plot(data_sizes, scratch_accuracies, "s--", label="From Scratch", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Data Efficiency: Fine-Tuning vs Training from Scratch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
