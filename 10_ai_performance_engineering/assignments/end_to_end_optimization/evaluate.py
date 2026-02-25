"""Benchmark suite, waterfall chart, and performance report generation.

Usage:
    python evaluate.py --benchmark-suite    # Run all configurations
    python evaluate.py --report             # Print report template
"""

import argparse
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import numpy as np


def print_report_template():
    """Print the performance engineering report template."""
    print("=" * 80)
    print("PERFORMANCE ENGINEERING REPORT")
    print("=" * 80)

    print("\n## 1. Executive Summary")
    print("  Baseline throughput: ___ samp/s")
    print("  Final throughput: ___ samp/s")
    print("  Total speedup: ___x")
    print("  Accuracy delta: ___")

    print("\n## 2. Speedup Attribution Table")
    steps = [
        "Baseline", "+ DataLoader opt", "+ Mixed precision",
        "+ torch.compile", "+ Flash Attention", "+ Grad checkpoint",
        "+ Batch size increase", "FINAL",
    ]
    header = f"{'Optimization':<25} {'Standalone':>12} {'Cumulative':>15} {'Speedup':>10}"
    print(header)
    print("-" * len(header))
    for step in steps:
        print(f"{step:<25} {'___':>12} {'___':>15} {'___':>10}")

    print("\n## 3. Accuracy Verification")
    print("  Baseline accuracy: ___")
    print("  Optimized accuracy: ___")
    print("  Delta: ___")

    print("\n## 4. Recommendations")
    print("  1. Always start with DataLoader optimization")
    print("  2. Mixed precision is nearly free -- always enable it")
    print("  3. torch.compile gives best results on compute-bound models")
    print("  4. Flash Attention helps most with long sequences")
    print("  5. Increase batch size only after freeing memory with other techniques")


def generate_waterfall_chart():
    """Generate a waterfall chart showing per-technique speedup contribution."""
    print("Generate waterfall chart after running the benchmark suite.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Optimization Technique")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Optimization Waterfall: Contribution of Each Technique")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="E2E Optimization Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--benchmark-suite", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        print_report_template()

    if args.benchmark_suite:
        print("Run each optimization configuration and collect metrics.")
        generate_waterfall_chart()


if __name__ == "__main__":
    main()
