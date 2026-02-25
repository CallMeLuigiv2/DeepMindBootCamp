"""Profile comparison and visualization for profiling assignment.

Usage:
    python evaluate.py --report          # Print before/after measurements table
    python evaluate.py --waterfall       # Generate waterfall chart of speedups
"""

import argparse
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import numpy as np


def print_report_template():
    """Print the profiling report template to fill in."""
    print("=" * 80)
    print("PROFILING REPORT")
    print("=" * 80)

    print("\n## Before/After Measurements\n")
    header = (
        f"{'Metric':<22} {'Baseline':>10} {'DataLoader':>10} {'Augment':>10} "
        f"{'NoSync':>10} {'BatchSize':>10} {'AMP':>10} {'Compile':>10} {'Final':>10}"
    )
    print(header)
    print("-" * len(header))
    metrics = [
        "Throughput (samp/s)",
        "GPU Utilization",
        "Peak Memory (MB)",
        "Data Load Time (ms)",
        "Forward Time (ms)",
        "Backward Time (ms)",
        "Optim Step (ms)",
    ]
    for m in metrics:
        print(f"{m:<22}" + " ___" * 8)

    print("\n## Speedup Waterfall")
    print("  Fill in the standalone and cumulative speedup for each optimization.\n")

    steps = [
        "Baseline",
        "+ DataLoader fix",
        "+ Augmentation fix",
        "+ Sync removal",
        "+ Batch size increase",
        "+ Mixed precision",
        "+ torch.compile",
    ]
    print(f"{'Optimization':<25} {'Standalone':>12} {'Cumulative':>12}")
    print("-" * 49)
    for step in steps:
        print(f"{step:<25} {'___':>12} {'___':>12}")


def generate_waterfall_chart():
    """Generate a waterfall chart from experiment data."""
    print("Generate waterfall chart after collecting measurements.")
    print("(Fill in the data from your experiments.)")

    # Placeholder
    fig, ax = plt.subplots(figsize=(10, 6))
    optimizations = [
        "Baseline", "DataLoader", "Augmentation",
        "Sync Remove", "Batch Size", "Mixed Prec.", "Compile"
    ]
    ax.set_xlabel("Optimization")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Optimization Waterfall Chart")
    ax.set_xticks(range(len(optimizations)))
    ax.set_xticklabels(optimizations, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Profiling Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--waterfall", action="store_true")
    args = parser.parse_args()

    if args.report:
        print_report_template()

    if args.waterfall:
        generate_waterfall_chart()


if __name__ == "__main__":
    main()
