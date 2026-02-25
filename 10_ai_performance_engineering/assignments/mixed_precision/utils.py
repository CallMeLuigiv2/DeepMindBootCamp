"""Memory tracking, timing utilities, and numerical precision exploration.

All functions in this file are fully implemented (no stubs).
"""

import os
import time
import tempfile
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


def measure_model_size(model: nn.Module) -> float:
    """Measure model size on disk in MB.

    Args:
        model: PyTorch model.

    Returns:
        Model size in MB.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
    os.unlink(f.name)
    return size_bytes / (1024 ** 2)


def measure_inference_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 20,
    use_cuda: bool = False,
) -> dict[str, float]:
    """Measure inference latency.

    Args:
        model: Model to benchmark.
        input_tensor: Sample input.
        num_runs: Number of timed runs.
        warmup_runs: Number of warmup runs.
        use_cuda: Whether to use CUDA timing.

    Returns:
        Dictionary with 'mean_ms', 'std_ms', 'min_ms', 'p99_ms'.
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if use_cuda and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(input_tensor)
                times.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "p99_ms": t.quantile(0.99).item(),
    }


def measure_gpu_memory(
    func,
    *args,
) -> dict[str, float]:
    """Measure peak GPU memory during a function call.

    Args:
        func: Function to measure.
        *args: Arguments to func.

    Returns:
        Dictionary with 'peak_mb', 'allocated_mb'.
    """
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "allocated_mb": 0.0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    func(*args)
    torch.cuda.synchronize()

    return {
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
    }


def explore_numerical_precision():
    """Demonstrate numerical properties of FP32, FP16, and BF16.

    Prints a comprehensive comparison table.
    """
    print("=" * 70)
    print("Numerical Precision Exploration")
    print("=" * 70)

    # Smallest positive normal number
    print("\n1. Smallest positive normal number:")
    formats = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }
    for name, dtype in formats.items():
        info = torch.finfo(dtype)
        print(f"  {name}: {info.tiny:.6e} (eps={info.eps:.6e})")

    # Precision loss: 1.0 + x
    print("\n2. Precision loss (1.0 + x):")
    for x_val in [1e-4, 1e-5, 1e-6]:
        print(f"  x = {x_val:.0e}:")
        for name, dtype in formats.items():
            one = torch.tensor(1.0, dtype=dtype)
            x = torch.tensor(x_val, dtype=dtype)
            result = one + x
            lost = (result - one).item() == 0
            print(f"    {name}: 1 + x = {result.item():.10f} {'(LOST!)' if lost else ''}")

    # Gradient underflow
    print("\n3. Gradient underflow (value = 1e-6):")
    for name, dtype in formats.items():
        val = torch.tensor(1e-6, dtype=torch.float32)
        cast = val.to(dtype)
        print(f"  {name}: {val.item():.6e} -> {cast.item():.6e} {'(underflow!)' if cast.item() == 0 else ''}")

    # Loss scaling demonstration
    print("\n4. Loss scaling (scale = 1024, value = 1e-6):")
    val = torch.tensor(1e-6, dtype=torch.float32)
    scaled = val * 1024
    for name, dtype in formats.items():
        cast = scaled.to(dtype)
        unscaled = (cast / 1024).float()
        print(f"  {name}: scaled={cast.item():.6e}, unscaled={unscaled.item():.6e}, preserved={abs(unscaled.item() - 1e-6) < 1e-7}")


def format_comparison_table(results: dict[str, dict]) -> str:
    """Format a comparison table as a string.

    Args:
        results: Dictionary mapping config name to metric dictionary.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to display."

    configs = list(results.keys())
    metrics = list(results[configs[0]].keys())

    # Header
    col_width = max(15, max(len(c) for c in configs) + 2)
    header = f"{'Metric':<25}" + "".join(f"{c:>{col_width}}" for c in configs)
    separator = "-" * len(header)

    lines = [header, separator]
    for metric in metrics:
        row = f"{metric:<25}"
        for config in configs:
            val = results[config].get(metric, "N/A")
            if isinstance(val, float):
                if val > 100:
                    row += f"{val:>{col_width}.1f}"
                else:
                    row += f"{val:>{col_width}.4f}"
            else:
                row += f"{str(val):>{col_width}}"
        lines.append(row)

    return "\n".join(lines)
