"""Profiling utilities: context managers, timeline export, bottleneck helpers.

All functions in this file are fully implemented (no stubs).
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class PhaseTimer:
    """Time individual phases of a training step.

    Usage:
        timer = PhaseTimer(use_cuda=True)
        timer.start("data_loading")
        batch = next(iter(loader))
        timer.stop("data_loading")
        timer.start("forward")
        output = model(batch)
        timer.stop("forward")
        ...
        timer.report()
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._start_times: dict[str, float] = {}
        self._durations: dict[str, list[float]] = {}

    def start(self, phase: str):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._start_times[phase] = time.perf_counter()

    def stop(self, phase: str):
        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_times[phase]
        self._durations.setdefault(phase, []).append(elapsed * 1000.0)

    def report(self, skip_first_n: int = 0):
        """Print a summary table of phase timings.

        Args:
            skip_first_n: Number of initial measurements to skip (warmup).
        """
        header = f"{'Phase':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'% Total':>10}"
        print(header)
        print("-" * len(header))

        total_mean = 0.0
        phase_means = {}
        for phase, times in self._durations.items():
            trimmed = times[skip_first_n:] if len(times) > skip_first_n else times
            if trimmed:
                t = torch.tensor(trimmed)
                mean_ms = t.mean().item()
                std_ms = t.std().item() if len(trimmed) > 1 else 0.0
                phase_means[phase] = mean_ms
                total_mean += mean_ms

        for phase, mean_ms in phase_means.items():
            trimmed = self._durations[phase][skip_first_n:]
            t = torch.tensor(trimmed)
            std_ms = t.std().item() if len(trimmed) > 1 else 0.0
            pct = 100.0 * mean_ms / total_mean if total_mean > 0 else 0
            print(f"{phase:<25} {mean_ms:>12.2f} {std_ms:>12.2f} {pct:>9.1f}%")

        print("-" * len(header))
        print(f"{'TOTAL':<25} {total_mean:>12.2f}")

    def get_means(self, skip_first_n: int = 0) -> dict[str, float]:
        """Return mean duration for each phase."""
        result = {}
        for phase, times in self._durations.items():
            trimmed = times[skip_first_n:] if len(times) > skip_first_n else times
            if trimmed:
                result[phase] = torch.tensor(trimmed).mean().item()
        return result


def setup_profiler(
    trace_dir: str = "traces",
    wait: int = 1,
    warmup: int = 2,
    active: int = 5,
    repeat: int = 1,
) -> torch.profiler.profile:
    """Create a configured torch.profiler.profile context manager.

    Args:
        trace_dir: Directory to save Chrome trace files.
        wait: Number of steps to wait before profiling.
        warmup: Number of warmup steps.
        active: Number of steps to actively profile.
        repeat: Number of profiling cycles.

    Returns:
        Configured profiler context manager.
    """
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat,
    )

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def measure_dataloader_throughput(
    dataloader,
    num_batches: int = 100,
    device: Optional[torch.device] = None,
) -> float:
    """Measure DataLoader throughput independently of model computation.

    Args:
        dataloader: The DataLoader to benchmark.
        num_batches: Number of batches to iterate.
        device: If provided, also transfer data to this device.

    Returns:
        Throughput in samples per second.
    """
    total_samples = 0
    start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        if device is not None:
            data = data.to(device, non_blocking=True)

        total_samples += data.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return total_samples / elapsed if elapsed > 0 else 0.0


def get_gpu_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def reset_memory_stats():
    """Reset GPU memory statistics for fresh measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def measure_throughput(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    num_steps: int = 50,
    warmup_steps: int = 10,
) -> float:
    """Measure training throughput in samples per second.

    Args:
        model: The model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        num_steps: Total number of steps.
        warmup_steps: Steps to skip for warmup.

    Returns:
        Throughput in samples per second.
    """
    model.train()
    total_samples = 0
    data_iter = iter(dataloader)

    # Warmup
    for _ in range(warmup_steps):
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data, target = next(data_iter)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_steps):
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data, target = next(data_iter)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_samples += data.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return total_samples / elapsed if elapsed > 0 else 0.0
