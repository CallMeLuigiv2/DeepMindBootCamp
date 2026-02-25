"""DDP initialization helpers and distributed training utilities.

All functions in this file are fully implemented (no stubs).
"""

import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize the distributed process group.

    Must be called before creating DDP models. Assumes torchrun was used
    to launch the script (environment variables set automatically).

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a tensor across all processes and average.

    Args:
        tensor: Tensor to reduce.
        world_size: Number of processes.

    Returns:
        Averaged tensor (same on all ranks after all-reduce).
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def verify_gradient_sync(model: nn.Module, world_size: int) -> bool:
    """Verify that gradients are synchronized across all DDP ranks.

    After a backward pass with DDP, all ranks should have identical gradients.
    This function checks that by all-reducing the sum of gradient differences.

    Args:
        model: The DDP-wrapped model.
        world_size: Number of processes.

    Returns:
        True if gradients are synchronized across all ranks.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_sum = param.grad.sum().clone()
            dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
            # After DDP backward, all ranks should have identical gradients
            # So grad_sum should be world_size * single_rank_grad_sum
            expected = param.grad.sum() * world_size
            if not torch.allclose(grad_sum, expected, atol=1e-5):
                return False
    return True


class DistributedMetricTracker:
    """Track and aggregate metrics across distributed processes.

    Usage:
        tracker = DistributedMetricTracker()
        tracker.update(loss=loss.item(), correct=num_correct, total=batch_size)
        # At end of epoch:
        metrics = tracker.compute()  # all-reduced across ranks
        tracker.reset()
    """

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, **kwargs: float):
        for key, value in kwargs.items():
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1

    def compute(self, world_size: int = 1) -> dict[str, float]:
        """Compute aggregated metrics, optionally across distributed ranks.

        Args:
            world_size: Number of processes for all-reduce.

        Returns:
            Dictionary of averaged metrics.
        """
        results = {}
        for key in self._sums:
            val = torch.tensor(self._sums[key], dtype=torch.float64)
            count = torch.tensor(self._counts[key], dtype=torch.float64)

            if dist.is_initialized() and world_size > 1:
                if val.is_cuda:
                    pass  # already on correct device
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    val = val.to(device)
                    count = count.to(device)
                dist.all_reduce(val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)

            results[key] = (val / count).item() if count.item() > 0 else 0.0

        return results

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class ThroughputTimer:
    """Measure training throughput (samples per second).

    Usage:
        timer = ThroughputTimer()
        for batch in dataloader:
            timer.start()
            train_step(batch)
            timer.stop(batch_size=len(batch))
        print(f"Throughput: {timer.throughput():.1f} samples/sec")
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._total_time: float = 0.0
        self._total_samples: int = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self, batch_size: int):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        self._total_time += elapsed
        self._total_samples += batch_size

    def throughput(self) -> float:
        """Return samples per second."""
        if self._total_time == 0:
            return 0.0
        return self._total_samples / self._total_time

    def reset(self):
        self._start_time = None
        self._total_time = 0.0
        self._total_samples = 0
