"""Evaluation, verification, and benchmarking for custom autograd functions.

Usage:
    python evaluate.py --verify-all       # Run all gradient checks
    python evaluate.py --benchmark        # Run performance benchmarks
    python evaluate.py --compare-ste      # Compare STE training curves
"""

import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt

from shared_utils.common import get_device

from model import (
    ParameterizedSwish,
    LearnableSwish,
    HardThresholdSTE,
    ClampedSTE,
    AsymmetricMSEFunction,
)
from utils import (
    verify_gradcheck,
    compare_outputs,
    compare_gradients,
    benchmark_function,
    measure_peak_memory,
    format_bytes,
    Timer,
)


def verify_parameterized_swish(device: torch.device) -> bool:
    """Verify ParameterizedSwish correctness and gradients."""
    print("\n--- Verifying ParameterizedSwish ---")

    x = torch.randn(16, requires_grad=True, device=device)
    beta = torch.tensor(1.5, requires_grad=True, device=device)

    # Output correctness
    custom_out = ParameterizedSwish.apply(x, beta)
    reference_out = x * torch.sigmoid(beta * x)
    out_ok = compare_outputs(custom_out, reference_out, "forward output")

    # Gradient correctness
    loss_custom = custom_out.sum()
    loss_custom.backward()
    grad_x_custom = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    beta2 = beta.detach().clone().requires_grad_(True)
    ref_out2 = x2 * torch.sigmoid(beta2 * x2)
    ref_out2.sum().backward()
    grad_x_ref = x2.grad

    grad_ok = compare_gradients(grad_x_custom, grad_x_ref, "grad_x")

    # Gradcheck with float64
    x_check = torch.randn(8, dtype=torch.float64, requires_grad=True, device=device)
    beta_check = torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device=device)
    gc_ok = verify_gradcheck(ParameterizedSwish.apply, (x_check, beta_check), "ParameterizedSwish")

    return out_ok and grad_ok and gc_ok


def verify_ste(device: torch.device) -> bool:
    """Verify STE implementations."""
    print("\n--- Verifying HardThresholdSTE ---")
    x = torch.randn(8, dtype=torch.float64, requires_grad=True, device=device)
    gc1 = verify_gradcheck(HardThresholdSTE.apply, (x,), "HardThresholdSTE")

    print("\n--- Verifying ClampedSTE ---")
    x2 = torch.randn(8, dtype=torch.float64, requires_grad=True, device=device)
    gc2 = verify_gradcheck(ClampedSTE.apply, (x2,), "ClampedSTE")

    return gc1 and gc2


def verify_asymmetric_mse(device: torch.device) -> bool:
    """Verify AsymmetricMSEFunction correctness and gradients."""
    print("\n--- Verifying AsymmetricMSEFunction ---")
    y_pred = torch.randn(16, dtype=torch.float64, requires_grad=True, device=device)
    y_true = torch.randn(16, dtype=torch.float64, device=device)

    gc_ok = verify_gradcheck(
        lambda yp, yt: AsymmetricMSEFunction.apply(yp, yt, 1.0, 2.0),
        (y_pred, y_true),
        "AsymmetricMSEFunction",
    )
    return gc_ok


def run_benchmarks(config: dict, device: torch.device) -> None:
    """Benchmark ParameterizedSwish vs pure-PyTorch implementation."""
    print("\n" + "=" * 60)
    print("Performance Benchmark: Custom vs Pure-PyTorch Swish")
    print("=" * 60)

    bench_cfg = config["benchmark"]
    use_cuda = device.type == "cuda"
    beta = torch.tensor(1.5, device=device, requires_grad=True)

    for size in bench_cfg["input_sizes"]:
        print(f"\nInput size: {size:,}")
        x = torch.randn(size, device=device, requires_grad=True)

        # Forward: custom
        def fwd_custom():
            return ParameterizedSwish.apply(x, beta)

        # Forward: reference
        def fwd_reference():
            return x * torch.sigmoid(beta * x)

        benchmark_function(
            fwd_custom, (), num_runs=bench_cfg["num_runs"],
            warmup_runs=bench_cfg["warmup_runs"], use_cuda=use_cuda,
            label="Custom forward",
        )
        benchmark_function(
            fwd_reference, (), num_runs=bench_cfg["num_runs"],
            warmup_runs=bench_cfg["warmup_runs"], use_cuda=use_cuda,
            label="PyTorch forward",
        )

        # Backward: custom
        def bwd_custom():
            out = ParameterizedSwish.apply(x.detach().requires_grad_(True), beta)
            out.sum().backward()

        def bwd_reference():
            x2 = x.detach().requires_grad_(True)
            out = x2 * torch.sigmoid(beta * x2)
            out.sum().backward()

        benchmark_function(
            bwd_custom, (), num_runs=bench_cfg["num_runs"],
            warmup_runs=bench_cfg["warmup_runs"], use_cuda=use_cuda,
            label="Custom fwd+bwd",
        )
        benchmark_function(
            bwd_reference, (), num_runs=bench_cfg["num_runs"],
            warmup_runs=bench_cfg["warmup_runs"], use_cuda=use_cuda,
            label="PyTorch fwd+bwd",
        )

        # Memory comparison
        if use_cuda:
            mem_custom = measure_peak_memory(fwd_custom, ())
            mem_ref = measure_peak_memory(fwd_reference, ())
            print(f"  Memory custom:  {format_bytes(mem_custom)}")
            print(f"  Memory PyTorch: {format_bytes(mem_ref)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom Autograd Functions")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--verify-all", action="store_true", help="Run all gradient checks")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--compare-ste", action="store_true", help="Plot STE training curves")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()

    if args.verify_all:
        print("=" * 60)
        print("Gradient Verification Suite")
        print("=" * 60)
        all_ok = True
        all_ok &= verify_parameterized_swish(device)
        all_ok &= verify_ste(device)
        all_ok &= verify_asymmetric_mse(device)
        print("\n" + "=" * 60)
        print(f"Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
        print("=" * 60)

    if args.benchmark:
        run_benchmarks(config, device)

    if args.compare_ste:
        print("Load training logs and plot STE comparison curves.")
        print("(Run train.py first to generate the logs.)")


if __name__ == "__main__":
    main()
