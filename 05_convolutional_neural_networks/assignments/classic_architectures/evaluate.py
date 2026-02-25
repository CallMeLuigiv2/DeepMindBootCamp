"""Evaluation and comparison for the Classic Architectures project.

Pre-written: model loading, gradient norm computation.
Stubbed: comparison visualization, per-class analysis.

Usage:
    python evaluate.py --checkpoint checkpoints/resnet18_best.pth --arch resnet18
    python evaluate.py --gradient-analysis --arch resnet18 --arch2 plainnet18
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import get_device, load_checkpoint, count_parameters
from shared_utils.plotting import plot_confusion_matrix

from model import create_model
from data import get_cifar10_loaders, CIFAR10_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Classic Architectures")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--arch2", type=str, default=None, help="Second arch for comparison")
    parser.add_argument("--gradient-analysis", action="store_true")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


# ============================================================
# Gradient Analysis (Pre-written for skip connection experiment)
# ============================================================

def check_gradient_norms(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute gradient norms at each layer.

    This is for Part 5 -- the skip connection experiment. Compare gradient
    norms between ResNet-18 and PlainNet-18 to see if gradients vanish
    in the plain network.

    Args:
        model: Trained model.
        dataloader: A DataLoader (one batch is enough).
        device: Device.

    Returns:
        Dictionary mapping layer names to gradient norms.
    """
    model.train()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_norms[name] = param.grad.norm().item()

    model.zero_grad()
    return grad_norms


def plot_gradient_comparison(
    grad_norms_resnet: dict[str, float],
    grad_norms_plainnet: dict[str, float],
    save_path: str = None,
) -> None:
    """Plot gradient norms for ResNet vs PlainNet side by side.

    Args:
        grad_norms_resnet: Gradient norms from ResNet-18.
        grad_norms_plainnet: Gradient norms from PlainNet-18.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (title, norms) in zip(axes, [
        ("ResNet-18", grad_norms_resnet),
        ("PlainNet-18", grad_norms_plainnet),
    ]):
        names = list(norms.keys())
        values = list(norms.values())
        ax.barh(range(len(names)), values, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Gradient Norm")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Comparison Visualization (Stubbed)
# ============================================================

def plot_training_curves_comparison(results: dict, save_path: str = None) -> None:
    """Plot training curves for all architectures on one plot.

    Args:
        results: Dict mapping arch names to their training histories.
        save_path: Path to save figure.
    """
    # YOUR CODE HERE
    # Hint: Load TensorBoard event files or saved training logs
    # Plot test accuracy vs epoch for all architectures with different colors
    pass


def plot_parameter_efficiency(results: dict, save_path: str = None) -> None:
    """Plot test accuracy vs parameter count.

    Shows which model gives the best accuracy per parameter.

    Args:
        results: Dict mapping arch names to dicts with 'params' and 'test_acc'.
        save_path: Path to save figure.
    """
    # YOUR CODE HERE
    pass


def compute_per_class_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class accuracy.

    Args:
        model: Trained model.
        loader: Test DataLoader.
        device: Device.
        num_classes: Number of classes.

    Returns:
        Tuple of (all_preds, all_labels, per_class_accuracy).
    """
    # YOUR CODE HERE
    pass


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Gradient analysis (Part 5)
    if args.gradient_analysis:
        print("Running gradient analysis: ResNet-18 vs PlainNet-18")

        _, _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

        # ResNet-18
        resnet = create_model("resnet18").to(device)
        ckpt_path = os.path.join("checkpoints", "resnet18_best.pth")
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, resnet)
        grad_norms_resnet = check_gradient_norms(resnet, test_loader, device)

        # PlainNet-18
        plainnet = create_model("plainnet18").to(device)
        ckpt_path = os.path.join("checkpoints", "plainnet18_best.pth")
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, plainnet)
        grad_norms_plain = check_gradient_norms(plainnet, test_loader, device)

        # Plot comparison
        plot_gradient_comparison(
            grad_norms_resnet, grad_norms_plain,
            save_path=os.path.join(args.output_dir, "gradient_comparison.png"),
        )

        print("\nResNet-18 gradient norms:")
        for name, norm in list(grad_norms_resnet.items())[:10]:
            print(f"  {name:40s}  {norm:.6f}")

        print("\nPlainNet-18 gradient norms:")
        for name, norm in list(grad_norms_plain.items())[:10]:
            print(f"  {name:40s}  {norm:.6f}")

        return

    # Standard evaluation
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading {args.arch} from {args.checkpoint}")
        model = create_model(args.arch).to(device)
        ckpt = load_checkpoint(args.checkpoint, model)
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Parameters: {count_parameters(model, trainable_only=False):,}")

        _, _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

        # Overall accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f"\n  Test Accuracy: {correct / total:.4f} ({correct}/{total})")

        # Per-class accuracy
        print("\n  Per-class accuracy:")
        result = compute_per_class_accuracy(model, test_loader, device)
        if result is not None:
            all_preds, all_labels, per_class_acc = result
            for i, name in enumerate(CIFAR10_CLASSES):
                print(f"    {name:<15} {per_class_acc[i]:.4f}")
    else:
        print("No checkpoint specified or file not found.")
        print("Usage: python evaluate.py --checkpoint checkpoints/resnet18_best.pth --arch resnet18")


if __name__ == "__main__":
    main()
