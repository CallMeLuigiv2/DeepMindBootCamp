"""Throughput measurement, memory budget, and benchmark harness.

All functions in this file are fully implemented (no stubs).
"""

import time
from typing import Optional, Callable

import torch
import torch.nn as nn


class BenchmarkHarness:
    """Run training benchmarks with consistent methodology.

    Handles warmup, timing, GPU synchronization, and metric collection.

    Usage:
        harness = BenchmarkHarness(model, dataloader, criterion, optimizer, device)
        results = harness.run(num_steps=100, warmup_steps=20)
        print(f"Throughput: {results['throughput']:.1f} samp/s")
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        optimizer,
        device: torch.device,
        scaler=None,
        autocast_dtype=None,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.autocast_dtype = autocast_dtype
        self.accumulation_steps = accumulation_steps

    def _train_step(self, data, target):
        """Execute a single training step."""
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        if self.autocast_dtype is not None:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                output = self.model(data)
                loss = self.criterion(output, target) / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            output = self.model(data)
            loss = self.criterion(output, target) / self.accumulation_steps
            loss.backward()

        return loss.item() * self.accumulation_steps, data.size(0)

    def run(
        self,
        num_steps: int = 100,
        warmup_steps: int = 20,
    ) -> dict[str, float]:
        """Run benchmark and collect metrics.

        Args:
            num_steps: Number of timed training steps.
            warmup_steps: Number of warmup steps (not timed).

        Returns:
            Dictionary with throughput, peak_memory_mb, avg_loss.
        """
        self.model.train()
        data_iter = iter(self.dataloader)

        def next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                return next(data_iter)

        # Warmup
        for _ in range(warmup_steps):
            self.optimizer.zero_grad()
            for _ in range(self.accumulation_steps):
                data, target = next_batch()
                self._train_step(data, target)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Timed run
        total_samples = 0
        total_loss = 0.0
        start = time.perf_counter()

        for step in range(num_steps):
            self.optimizer.zero_grad()
            for _ in range(self.accumulation_steps):
                data, target = next_batch()
                loss_val, batch_size = self._train_step(data, target)
                total_loss += loss_val
                total_samples += batch_size
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

        return {
            "throughput": total_samples / elapsed,
            "peak_memory_mb": peak_mem,
            "avg_loss": total_loss / num_steps,
            "total_time_s": elapsed,
            "total_samples": total_samples,
        }


def compute_memory_budget(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """Compute estimated memory budget in MB.

    Args:
        num_params: Total number of model parameters.
        batch_size: Batch size.
        seq_length: Sequence length.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        dtype_bytes: Bytes per element (4 for FP32, 2 for FP16).

    Returns:
        Dictionary with memory estimates in MB.
    """
    param_mem = num_params * dtype_bytes / (1024 ** 2)
    grad_mem = param_mem  # Same size as parameters
    optimizer_mem = num_params * 4 * 2 / (1024 ** 2)  # AdamW: 2 states in FP32

    # Activation memory estimate: per-layer activations
    # Each layer stores: input (B, S, H), attention (B, H, S, S), FFN intermediate (B, S, 4H)
    per_layer_activation = (
        batch_size * seq_length * hidden_dim * dtype_bytes  # Input
        + batch_size * 8 * seq_length * seq_length * dtype_bytes  # Attention (approx)
        + batch_size * seq_length * hidden_dim * 4 * dtype_bytes  # FFN
    ) / (1024 ** 2)
    activation_mem = per_layer_activation * num_layers

    return {
        "parameters_mb": param_mem,
        "gradients_mb": grad_mem,
        "optimizer_mb": optimizer_mem,
        "activations_mb": activation_mem,
        "total_mb": param_mem + grad_mem + optimizer_mem + activation_mem,
    }


def print_benchmark_results(results: dict[str, dict], title: str = "Benchmark Results"):
    """Print formatted benchmark results table.

    Args:
        results: Dictionary mapping config names to metric dictionaries.
        title: Table title.
    """
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    configs = list(results.keys())
    metrics = ["throughput", "peak_memory_mb", "avg_loss", "total_time_s"]

    header = f"{'Config':<30}" + "".join(f"{m:>15}" for m in metrics)
    print(header)
    print("-" * len(header))

    for config in configs:
        row = f"{config:<30}"
        for metric in metrics:
            val = results[config].get(metric, 0)
            if metric == "throughput":
                row += f"{val:>15.1f}"
            elif metric == "peak_memory_mb":
                row += f"{val:>15.1f}"
            else:
                row += f"{val:>15.4f}"
        print(row)

    # Speedup relative to first config
    if len(configs) > 1:
        base_throughput = results[configs[0]]["throughput"]
        print()
        print("Speedup vs baseline:")
        for config in configs:
            speedup = results[config]["throughput"] / base_throughput if base_throughput > 0 else 0
            print(f"  {config}: {speedup:.2f}x")
