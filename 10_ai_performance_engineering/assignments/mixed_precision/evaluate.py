"""Quantization, benchmarking, comparison tables, and Pareto frontier.

Usage:
    python evaluate.py --quantize          # Apply dynamic and static quantization
    python evaluate.py --benchmark         # Benchmark all configurations
    python evaluate.py --numerical         # Run numerical precision exploration
    python evaluate.py --compare-all       # Generate comprehensive comparison table
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

from shared_utils.common import get_device

from model import ResNet18ForQuantization
from data import load_cifar10_standard
from utils import (
    measure_model_size,
    measure_inference_latency,
    explore_numerical_precision,
    format_comparison_table,
)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to a model.

    Args:
        model: Trained FP32 model.

    Returns:
        Dynamically quantized model.
    """
    # YOUR CODE HERE
    # Use torch.quantization.quantize_dynamic to quantize Linear layers to qint8
    # Return the quantized model
    raise NotImplementedError("Implement dynamic quantization")


def apply_static_quantization(
    model: nn.Module,
    calibration_loader,
    num_calibration_batches: int = 100,
) -> nn.Module:
    """Apply static INT8 quantization with calibration.

    Args:
        model: Trained FP32 model.
        calibration_loader: DataLoader for calibration data.
        num_calibration_batches: Number of batches for calibration.

    Returns:
        Statically quantized model.
    """
    # YOUR CODE HERE
    # 1. Set model to eval mode
    # 2. Prepare model for static quantization (torch.quantization.quantize_fx or manual)
    # 3. Run calibration data through the model
    # 4. Convert to quantized model
    raise NotImplementedError("Implement static quantization")


def benchmark_all_configs(config: dict) -> None:
    """Benchmark all precision configurations and print comparison table."""
    print("=" * 60)
    print("Benchmark: FP32 vs FP16 vs BF16 vs Dynamic INT8 vs Static INT8")
    print("=" * 60)
    print()
    print("Run train.py with each precision first, then run quantization.")
    print("Fill in the comprehensive comparison table with your measurements.")

    # Template
    header = (
        f"{'Configuration':<22} {'Train Spd':>10} {'Infer Spd':>10} "
        f"{'Latency':>10} {'Peak Mem':>10} {'Model MB':>10} {'Acc':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    configs = [
        "FP32 (baseline)",
        "FP16 mixed prec.",
        "BF16 mixed prec.",
        "FP16 inference",
        "Dynamic INT8 (CPU)",
        "Static INT8 (CPU)",
    ]
    for cfg in configs:
        print(f"{cfg:<22}" + " ___" * 6)


def plot_pareto_frontier() -> None:
    """Plot the Pareto frontier: latency vs accuracy."""
    print("\nGenerate Pareto frontier plot after collecting all measurements.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Inference Latency (ms/sample)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Precision-Performance Pareto Frontier")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Mixed Precision Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.numerical:
        explore_numerical_precision()

    if args.quantize:
        print("Apply quantization to trained models.")
        print("Load the FP32 checkpoint and apply dynamic/static quantization.")

    if args.benchmark:
        benchmark_all_configs(config)

    if args.compare_all:
        benchmark_all_configs(config)
        plot_pareto_frontier()


if __name__ == "__main__":
    main()
