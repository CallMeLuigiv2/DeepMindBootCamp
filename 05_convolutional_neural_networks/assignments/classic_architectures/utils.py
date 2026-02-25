"""Utility functions for the Classic Architectures project.

100% pre-written -- no stubs. Includes parameter counting helpers,
architecture verification, and training utilities.
"""

import sys
from pathlib import Path

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
from shared_utils.plotting import plot_training_curves


# ============================================================
# Parameter Counting Helpers
# ============================================================

def count_conv_parameters(model: nn.Module) -> int:
    """Count parameters in convolutional layers only."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total += sum(p.numel() for p in module.parameters())
    return total


def count_fc_parameters(model: nn.Module) -> int:
    """Count parameters in fully connected (Linear) layers only."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += sum(p.numel() for p in module.parameters())
    return total


def count_bn_parameters(model: nn.Module) -> int:
    """Count parameters in BatchNorm layers."""
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            total += sum(p.numel() for p in module.parameters())
    return total


def print_parameter_breakdown(model: nn.Module, model_name: str = "Model") -> None:
    """Print a detailed parameter breakdown.

    Args:
        model: The model to analyze.
        model_name: Name for display.
    """
    total = count_parameters(model, trainable_only=False)
    conv = count_conv_parameters(model)
    fc = count_fc_parameters(model)
    bn = count_bn_parameters(model)
    other = total - conv - fc - bn

    print(f"\n{model_name} Parameter Breakdown:")
    print(f"  Conv layers:      {conv:>12,} ({conv / total * 100:5.1f}%)")
    print(f"  FC layers:        {fc:>12,} ({fc / total * 100:5.1f}%)")
    print(f"  BatchNorm layers: {bn:>12,} ({bn / total * 100:5.1f}%)")
    if other > 0:
        print(f"  Other:            {other:>12,} ({other / total * 100:5.1f}%)")
    print(f"  Total:            {total:>12,}")


# ============================================================
# Architecture Verification
# ============================================================

def verify_output_shape(model: nn.Module, input_shape: tuple, expected_output: tuple) -> bool:
    """Verify that a model produces the expected output shape.

    Args:
        model: The model to test.
        input_shape: Input tensor shape (including batch dim).
        expected_output: Expected output shape (including batch dim).

    Returns:
        True if shapes match, False otherwise.
    """
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape).to(device)
    with torch.no_grad():
        output = model(dummy)

    if output.shape == expected_output:
        print(f"  Shape OK: {input_shape} -> {output.shape}")
        return True
    else:
        print(f"  Shape MISMATCH: {input_shape} -> {output.shape} (expected {expected_output})")
        return False


def verify_all_architectures() -> None:
    """Run shape verification on all architectures."""
    from model import create_model

    checks = [
        ("lenet5", (2, 3, 32, 32), (2, 10)),
        ("vgg11", (2, 3, 32, 32), (2, 10)),
        ("resnet18", (2, 3, 32, 32), (2, 10)),
        ("plainnet18", (2, 3, 32, 32), (2, 10)),
    ]

    print("Architecture Verification:")
    for arch, in_shape, out_shape in checks:
        print(f"\n{arch}:")
        model = create_model(arch)
        verify_output_shape(model, in_shape, torch.Size(out_shape))
        print_parameter_breakdown(model, arch)


# ============================================================
# Training History Helpers
# ============================================================

def save_training_history(history: dict, path: str) -> None:
    """Save training history to a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(path: str) -> dict:
    """Load training history from a JSON file."""
    import json
    with open(path) as f:
        return json.load(f)


def plot_comparison_curves(
    histories: dict[str, dict],
    metric: str = "val_acc",
    title: str = "Architecture Comparison",
    save_path: str = None,
) -> None:
    """Plot a single metric across multiple architectures.

    Args:
        histories: Dict mapping arch names to their training histories.
            Each history should have lists keyed by metric name.
        metric: Metric to plot (e.g., 'val_acc', 'val_loss').
        title: Plot title.
        save_path: Path to save figure.
    """
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (arch, history), color in zip(histories.items(), colors):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            plt.plot(epochs, history[metric], label=arch, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
