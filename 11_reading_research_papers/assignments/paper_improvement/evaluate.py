"""Statistical comparison and ablation study for paper improvement.

Usage:
    python evaluate.py --compare --statistical
    python evaluate.py --ablation
"""

import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from shared_utils.common import get_device

from utils import statistical_comparison, print_comparison_report, AblationStudy


def run_statistical_comparison() -> None:
    """Run statistical comparison between baseline and improved results."""
    print("=" * 60)
    print("Statistical Comparison")
    print("=" * 60)
    print()
    print("Load results from A/B comparison (train.py --ab-comparison)")
    print("and run statistical tests.")
    print()

    # Example (fill in with your actual results):
    baseline_accs = []   # e.g., [0.9125, 0.9108, 0.9132]
    improved_accs = []   # e.g., [0.9189, 0.9175, 0.9201]

    if baseline_accs and improved_accs:
        result = statistical_comparison(baseline_accs, improved_accs, "test_accuracy")
        print_comparison_report(result)
    else:
        print("No results available. Run train.py --ab-comparison first.")


def run_ablation(config: dict, device: torch.device) -> None:
    """Run ablation study to isolate the effect of each component."""
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    ablation = AblationStudy("Improvement Ablation")

    # YOUR CODE HERE
    # Add ablation configurations:
    # 1. Baseline (no changes)
    # 2. Your improvement alone
    # 3. If your improvement has multiple components, test each
    # 4. All components together

    print("Design and run your ablation experiments.")
    print("Use the AblationStudy class to track results.")

    ablation.report()


def plot_training_curves():
    """Plot baseline vs improved training curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss: Baseline vs Improved")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Test Accuracy: Baseline vs Improved")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("A/B Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Paper Improvement Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--statistical", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()

    if args.compare:
        if args.statistical:
            run_statistical_comparison()
        plot_training_curves()

    if args.ablation:
        run_ablation(config, device)


if __name__ == "__main__":
    main()
