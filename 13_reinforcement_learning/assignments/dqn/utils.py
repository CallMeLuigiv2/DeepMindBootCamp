"""
DQN Utilities
==============

Fully implemented helper functions for DQN training:
- Episode reward tracking and statistics
- Epsilon schedule computation
- Moving average calculations
- Seed setting and device selection
- Checkpoint save/load
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Episode Tracker
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Tracks episode rewards and computes training statistics.

    Maintains a history of episode rewards and provides methods for
    computing rolling averages, best performance windows, etc.
    """

    def __init__(self):
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.q_values: List[float] = []

    def add_episode(
        self,
        reward: float,
        length: int = 0,
        loss: float = 0.0,
        epsilon: float = 0.0,
        mean_q: float = 0.0,
    ) -> None:
        """Record metrics for a completed episode."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.q_values.append(mean_q)

    def moving_average(self, window: int = 100) -> float:
        """Compute moving average of the last `window` episode rewards."""
        if not self.rewards:
            return 0.0
        recent = self.rewards[-window:]
        return sum(recent) / len(recent)

    def best_average(self, window: int = 100) -> float:
        """Compute the best moving average achieved during training."""
        if len(self.rewards) < window:
            return sum(self.rewards) / max(len(self.rewards), 1)
        best = float("-inf")
        for i in range(len(self.rewards) - window + 1):
            avg = sum(self.rewards[i : i + window]) / window
            best = max(best, avg)
        return best

    def get_smoothed_rewards(self, window: int = 50) -> List[float]:
        """Return a smoothed reward curve using a rolling average.

        Each point is the mean of rewards in a window centered (or
        left-aligned for early episodes) on that episode index.
        """
        smoothed = []
        for i in range(len(self.rewards)):
            start = max(0, i - window + 1)
            smoothed.append(sum(self.rewards[start : i + 1]) / (i - start + 1))
        return smoothed

    def is_solved(self, threshold: float = 475.0, window: int = 100) -> bool:
        """Check if the environment is solved (avg reward >= threshold)."""
        if len(self.rewards) < window:
            return False
        return self.moving_average(window) >= threshold

    def summary(self, last_n: int = 100) -> str:
        """Return a formatted summary string."""
        recent = self.rewards[-last_n:] if self.rewards else [0.0]
        return (
            f"Episodes: {len(self.rewards)} | "
            f"Avg({last_n}): {sum(recent)/len(recent):.1f} | "
            f"Best avg({last_n}): {self.best_average(last_n):.1f} | "
            f"Last: {self.rewards[-1]:.1f}" if self.rewards else "No episodes"
        )


# ---------------------------------------------------------------------------
# Epsilon Schedule
# ---------------------------------------------------------------------------

def compute_epsilon(
    episode: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
) -> float:
    """Compute epsilon for a given episode using exponential decay.

    epsilon = max(epsilon_end, epsilon_start * epsilon_decay^episode)

    Args:
        episode: Current episode number (0-indexed).
        epsilon_start: Initial epsilon value.
        epsilon_end: Minimum epsilon value.
        epsilon_decay: Multiplicative decay factor per episode.

    Returns:
        Current epsilon value.
    """
    return max(epsilon_end, epsilon_start * (epsilon_decay ** episode))


def compute_epsilon_linear(
    step: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    decay_steps: int = 10000,
) -> float:
    """Compute epsilon using linear decay over a fixed number of steps.

    Args:
        step: Current step number.
        epsilon_start: Initial epsilon value.
        epsilon_end: Final epsilon value.
        decay_steps: Number of steps over which to decay.

    Returns:
        Current epsilon value.
    """
    fraction = min(1.0, step / decay_steps)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


# ---------------------------------------------------------------------------
# Moving Average
# ---------------------------------------------------------------------------

def moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Compute a simple moving average over a list of values.

    Uses a causal (left-aligned) window so that each output depends only
    on current and past values.

    Args:
        values: List of scalar values.
        window: Size of the averaging window.

    Returns:
        Numpy array of the same length as values with the smoothed curve.
    """
    if not values:
        return np.array([])
    smoothed = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(values[start : i + 1])
    return smoothed


# ---------------------------------------------------------------------------
# Seed and Device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available torch device (CUDA > CPU).

    Returns:
        torch.device for CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint Save / Load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    config: dict,
    filepath: str,
    **extra_data,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        model: The Q-network to save.
        optimizer: The optimizer state to save.
        episode: Current episode number.
        config: Training configuration dictionary.
        filepath: Path where the checkpoint file will be written.
        **extra_data: Any additional data to include in the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "config": config,
        "hidden_dim": config.get("hidden_dim", 128),
    }
    checkpoint.update(extra_data)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load a training checkpoint from disk.

    Args:
        filepath: Path to the checkpoint file.
        device: Torch device to map tensors to.

    Returns:
        Dictionary containing all checkpoint data.
    """
    return torch.load(filepath, map_location=device, weights_only=False)
