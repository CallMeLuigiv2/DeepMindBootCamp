"""Ablation study and evaluation for distributed training techniques.

Usage:
    python evaluate.py --ablation     # Run ablation: disable techniques one at a time
    python evaluate.py --compare      # Compare single-GPU vs DDP vs full pipeline
"""

import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from shared_utils.common import get_device


def run_ablation(config: dict) -> None:
    """Ablation study: measure contribution of each technique.

    Disables each technique one at a time and measures:
    - Training throughput (samples/sec)
    - Peak GPU memory
    - Test accuracy
    """
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)
    print()
    print("Run the following configurations and record metrics:")
    print()
    print("1. All techniques enabled (full pipeline)")
    print("2. Disable torch.compile")
    print("3. Disable mixed precision")
    print("4. Disable gradient checkpointing")
    print("5. Disable gradient accumulation")
    print()

    # Placeholder table
    configs = [
        "Full pipeline",
        "- torch.compile",
        "- Mixed precision",
        "- Grad checkpointing",
        "- Grad accumulation",
    ]

    header = f"{'Configuration':<25} {'Throughput':>12} {'Peak Mem':>10} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for cfg_name in configs:
        print(f"{cfg_name:<25} {'___':>12} {'___':>10} {'___':>10}")

    print()
    print("Fill in measurements from your experiments.")


def compare_results(config: dict) -> None:
    """Compare training modes: single-GPU vs DDP vs full."""
    print("Load checkpoints and compare final accuracy across modes.")
    print("(Run train.py with each mode first.)")


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.ablation:
        run_ablation(config)

    if args.compare:
        compare_results(config)


if __name__ == "__main__":
    main()
