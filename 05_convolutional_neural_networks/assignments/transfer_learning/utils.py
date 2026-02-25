"""Utility functions for the Transfer Learning project.

100% pre-written -- no stubs. Includes image display helpers,
results logging, and experiment comparison utilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import (
    set_seed,
    get_device,
    count_parameters,
    print_model_summary,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
)


# ============================================================
# Results Logging
# ============================================================

class ExperimentLogger:
    """Logger for comparing transfer learning experiments.

    Usage:
        logger = ExperimentLogger()
        logger.log_strategy("frozen", val_acc=0.82, test_acc=0.80, time=120)
        logger.log_strategy("partial", val_acc=0.88, test_acc=0.87, time=180)
        logger.save("experiment_results.json")
        logger.print_comparison()
    """

    def __init__(self):
        self.experiments: dict[str, dict[str, Any]] = {}

    def log_strategy(self, name: str, **metrics) -> None:
        self.experiments[name] = metrics

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.experiments = json.load(f)

    def print_comparison(self) -> None:
        if not self.experiments:
            print("No experiments logged.")
            return

        metrics = set()
        for exp in self.experiments.values():
            metrics.update(exp.keys())
        metrics = sorted(metrics)

        header = f"{'Strategy':<15}" + "".join(f"{m:<15}" for m in metrics)
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        for name, exp in self.experiments.items():
            row = f"{name:<15}"
            for m in metrics:
                val = exp.get(m, "")
                if isinstance(val, float):
                    row += f"{val:<15.4f}"
                else:
                    row += f"{str(val):<15}"
            print(row)
        print("=" * len(header))


# ============================================================
# Image Display Helpers
# ============================================================

def show_dataset_samples(
    dataset,
    class_names: list[str],
    n_per_class: int = 3,
    n_classes: int = 10,
    save_path: str = None,
) -> None:
    """Display sample images from the dataset.

    Args:
        dataset: torchvision dataset.
        class_names: List of class names.
        n_per_class: Number of images per class to show.
        n_classes: Number of classes to display.
        save_path: Path to save figure.
    """
    from data import unnormalize

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.5, n_classes * 2.5),
    )

    # Collect images per class
    class_images: dict[int, list] = {}
    for img, label in dataset:
        label_int = label if isinstance(label, int) else label.item()
        if label_int not in class_images:
            class_images[label_int] = []
        if len(class_images[label_int]) < n_per_class:
            class_images[label_int].append(img)
        if all(len(v) >= n_per_class for v in list(class_images.values())[:n_classes]):
            break

    for row, cls in enumerate(sorted(class_images.keys())[:n_classes]):
        for col, img in enumerate(class_images[cls][:n_per_class]):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            img_np = unnormalize(img)
            ax.imshow(img_np)
            ax.axis("off")
            if col == 0:
                name = class_names[cls] if cls < len(class_names) else str(cls)
                ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=50)

    plt.suptitle("Dataset Samples", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_augmentation_comparison(
    original,
    transforms_dict: dict,
    n_images: int = 4,
    save_path: str = None,
) -> None:
    """Show the same images under different augmentation pipelines.

    Args:
        original: Original PIL images (list).
        transforms_dict: Dict mapping pipeline name to transform.
        n_images: Number of images to show.
        save_path: Path to save figure.
    """
    from data import unnormalize

    n_pipelines = len(transforms_dict)
    fig, axes = plt.subplots(n_pipelines, n_images, figsize=(n_images * 3, n_pipelines * 3))

    for row, (name, transform) in enumerate(transforms_dict.items()):
        for col in range(min(n_images, len(original))):
            img = transform(original[col])
            img_np = unnormalize(img)
            ax = axes[row][col] if n_pipelines > 1 else axes[col]
            ax.imshow(img_np)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=60)

    plt.suptitle("Augmentation Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Strategy Comparison Visualization
# ============================================================

def plot_strategy_comparison(
    results: dict[str, dict],
    metric: str = "test_acc",
    save_path: str = None,
) -> None:
    """Bar chart comparing strategies on a given metric.

    Args:
        results: Dict mapping strategy names to result dicts.
        metric: Metric to compare.
        save_path: Path to save figure.
    """
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors[:len(names)])

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Transfer Learning Strategy Comparison: {metric.replace('_', ' ').title()}")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
