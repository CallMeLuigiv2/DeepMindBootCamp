"""Visualization and evaluation for hooks and debugging experiments.

Usage:
    python evaluate.py --visualize-gradients  # Compare gradient flow plots
    python evaluate.py --visualize-activations  # Show activation monitor dashboard
    python evaluate.py --pruning-sweep  # Plot pruning ratio vs accuracy
"""

import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from shared_utils.common import get_device, set_seed

from model import DeepNetwork, SimpleMNISTNet
from data import load_mnist_flat
from utils import GradientFlowVisualizer, ActivationMonitor, MagnitudePruner


def visualize_gradient_flow(config: dict, device: torch.device) -> None:
    """Create side-by-side gradient flow plots for different network configurations."""
    print("Generating gradient flow comparison plots...")

    gf_cfg = config["gradient_flow"]
    train_loader, _, _ = load_mnist_flat(batch_size=64)

    configs = [
        ("Deep Sigmoid", {"activation": "sigmoid", "use_batchnorm": False, "use_residual": False}),
        ("Sigmoid + BN", {"activation": "sigmoid", "use_batchnorm": True, "use_residual": False}),
        ("Deep ReLU", {"activation": "relu", "use_batchnorm": False, "use_residual": False}),
        ("ReLU + Residual", {"activation": "relu", "use_batchnorm": False, "use_residual": True}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, cfg) in enumerate(configs):
        set_seed(42)
        model = DeepNetwork(
            input_size=784, hidden_size=gf_cfg["hidden_size"],
            num_layers=gf_cfg["num_layers"], num_classes=10, **cfg,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        viz = GradientFlowVisualizer(model)

        model.train()
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()

        if viz.grad_stats:
            layer_names = list(viz.grad_stats.keys())
            means = [viz.grad_stats[n]["mean"] for n in layer_names]
            maxes = [viz.grad_stats[n]["max"] for n in layer_names]

            ax = axes[idx]
            x = np.arange(len(layer_names))
            ax.bar(x, maxes, alpha=0.3, color="c", label="Max")
            ax.bar(x, means, alpha=0.7, color="b", label="Mean")
            ax.set_xticks(x[::max(1, len(x) // 10)])
            ax.set_xticklabels([layer_names[i] for i in range(0, len(layer_names), max(1, len(layer_names) // 10))],
                               rotation=45, ha="right", fontsize=6)
            ax.set_title(name)
            ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        viz.close()

    fig.suptitle("Gradient Flow Comparison", fontsize=14)
    plt.tight_layout()

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_dir / "gradient_flow_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved gradient flow comparison plot.")


def visualize_pruning_curve(config: dict, device: torch.device) -> None:
    """Plot pruning ratio vs test accuracy."""
    print("Generating pruning accuracy curve...")
    print("(Run train.py --experiment pruning first to generate the data.)")

    # Placeholder plot structure
    ratios = [0.0] + config["pruning"]["prune_ratios"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Magnitude Pruning: Accuracy vs Sparsity")
    ax.set_xticks(ratios)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Hooks and Debugging Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--visualize-gradients", action="store_true")
    parser.add_argument("--visualize-activations", action="store_true")
    parser.add_argument("--pruning-sweep", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()

    if args.visualize_gradients:
        visualize_gradient_flow(config, device)

    if args.visualize_activations:
        print("Run train.py first, then load captured activation stats.")

    if args.pruning_sweep:
        visualize_pruning_curve(config, device)


if __name__ == "__main__":
    main()
