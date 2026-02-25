"""Gradient checking and benchmarking utilities for custom autograd functions.

All functions in this file are fully implemented (no stubs).
"""

import time
from typing import Callable, Optional

import torch
import torch.nn as nn


def verify_gradcheck(
    func: Callable,
    inputs: tuple[torch.Tensor, ...],
    func_name: str = "Function",
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> bool:
    """Run torch.autograd.gradcheck on a custom function with float64 inputs.

    Args:
        func: The function to check (e.g., MyFunction.apply).
        inputs: Tuple of input tensors. Will be converted to float64
            with requires_grad=True.
        func_name: Name for logging.
        eps: Perturbation size for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        True if gradcheck passes, False otherwise.
    """
    # Convert to float64 and enable gradients
    double_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.is_floating_point():
            double_inputs.append(
                inp.detach().double().requires_grad_(True)
            )
        else:
            double_inputs.append(inp)

    try:
        result = torch.autograd.gradcheck(
            func,
            tuple(double_inputs),
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        print(f"  [PASS] gradcheck for {func_name}")
        return result
    except RuntimeError as e:
        print(f"  [FAIL] gradcheck for {func_name}: {e}")
        return False


def compare_outputs(
    output_custom: torch.Tensor,
    output_reference: torch.Tensor,
    name: str = "output",
    atol: float = 1e-6,
) -> bool:
    """Compare custom function output against a reference implementation.

    Args:
        output_custom: Output from custom autograd function.
        output_reference: Output from reference (pure-PyTorch) implementation.
        name: Label for logging.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    max_diff = (output_custom - output_reference).abs().max().item()
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max absolute difference = {max_diff:.2e} (tol={atol:.0e})")
    return passed


def compare_gradients(
    grad_custom: torch.Tensor,
    grad_reference: torch.Tensor,
    name: str = "gradient",
    atol: float = 1e-5,
) -> bool:
    """Compare gradients from custom function against reference.

    Args:
        grad_custom: Gradient from custom backward.
        grad_reference: Gradient from PyTorch autograd.
        name: Label for logging.
        atol: Absolute tolerance.

    Returns:
        True if gradients match within tolerance.
    """
    max_diff = (grad_custom - grad_reference).abs().max().item()
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max absolute difference = {max_diff:.2e} (tol={atol:.0e})")
    return passed


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("forward pass") as t:
            output = model(x)
        print(f"Time: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str = "", use_cuda: bool = False):
        self.name = name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


def benchmark_function(
    func: Callable,
    args: tuple,
    num_runs: int = 100,
    warmup_runs: int = 10,
    use_cuda: bool = False,
    label: str = "",
) -> dict[str, float]:
    """Benchmark a function's execution time over multiple runs.

    Args:
        func: Function to benchmark.
        args: Arguments to pass to the function.
        num_runs: Number of timed runs.
        warmup_runs: Number of warmup runs (not timed).
        use_cuda: Use CUDA events for GPU timing.
        label: Label for logging.

    Returns:
        Dictionary with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    # Warmup
    for _ in range(warmup_runs):
        func(*args)

    # Timed runs
    times = []
    for _ in range(num_runs):
        with Timer(use_cuda=use_cuda) as t:
            func(*args)
        times.append(t.elapsed_ms)

    times_tensor = torch.tensor(times)
    result = {
        "mean_ms": times_tensor.mean().item(),
        "std_ms": times_tensor.std().item(),
        "min_ms": times_tensor.min().item(),
        "max_ms": times_tensor.max().item(),
    }

    if label:
        print(
            f"  {label}: {result['mean_ms']:.3f} +/- {result['std_ms']:.3f} ms "
            f"(min={result['min_ms']:.3f}, max={result['max_ms']:.3f})"
        )

    return result


def measure_peak_memory(
    func: Callable,
    args: tuple,
) -> int:
    """Measure peak GPU memory allocated during a function call.

    Args:
        func: Function to measure.
        args: Arguments to pass to the function.

    Returns:
        Peak memory in bytes. Returns 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    func(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def format_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024**2:.1f} MB"
    else:
        return f"{n_bytes / 1024**3:.2f} GB"
