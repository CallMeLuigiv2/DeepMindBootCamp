"""Evaluation, results comparison, and ablation study.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --compare          # Compare results to paper
    python evaluate.py --ablation         # Run ablation study
"""

import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt

from shared_utils.common import get_device

from utils import ExperimentLogger, compare_results, count_parameters


def compare_to_paper(config: dict) -> None:
    """Compare our reproduction results to the paper's reported numbers."""
    print("=" * 60)
    print("Results Comparison: Paper vs Our Reproduction")
    print("=" * 60)

    # Example for ResNet paper
    paper_results = {
        "ResNet-20": 0.9125,   # Paper Table 6
        "ResNet-32": 0.9237,
        "ResNet-56": 0.9303,
        "Plain-20": 0.9149,
        "Plain-32": 0.9046,   # Degradation: worse than Plain-20
        "Plain-56": 0.8970,   # Further degradation
    }

    our_results = {}  # Fill in from experiment results

    print("\nPaper's CIFAR-10 Test Error Rates (Table 6):")
    print("(You need to match the experiment scale and report your numbers)")
    print()
    print(compare_results(our_results, paper_results, "Test Accuracy"))
    print()
    print("Note: Paper reports error rates; we report accuracy = 1 - error.")
    print("Document any discrepancies and hypothesize causes.")


def run_ablation(config: dict, device: torch.device) -> None:
    """Run ablation study: remove/change one component and measure effect."""
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)
    print()
    print("Design your ablation based on the paper you reproduced.")
    print("At minimum, test one of:")
    print("  - Remove the key contribution (e.g., remove skip connections)")
    print("  - Change a design choice (e.g., different shortcut type)")
    print("  - Vary a hyperparameter (e.g., network depth)")
    print()

    # YOUR CODE HERE
    # Run ablation experiments and record results
    print("Implement and run your ablation experiments here.")


def main():
    parser = argparse.ArgumentParser(description="Paper Reproduction Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()

    if args.compare:
        compare_to_paper(config)

    if args.ablation:
        run_ablation(config, device)


if __name__ == "__main__":
    main()
