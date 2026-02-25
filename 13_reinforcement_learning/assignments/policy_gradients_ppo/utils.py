"""
Policy Gradient Utilities
==========================

Fully implemented helper functions for policy gradient training:
- Episode tracking and statistics
- Advantage computation helpers
- Policy entropy calculation
- Explained variance metric
- Seed, device, and checkpoint utilities
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
    """Tracks episode rewards and training metrics across time.

    Maintains histories of rewards, losses, and other metrics. Provides
    methods for computing rolling averages and checking convergence.
    """

    def __init__(self):
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
        self.clip_fractions: List[float] = []
        self.timesteps: List[int] = []
        self.total_timesteps: int = 0

    def add_episode(
        self,
        reward: float,
        length: int = 0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
        clip_fraction: float = 0.0,
    ) -> None:
        """Record metrics for a completed episode."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.clip_fractions.append(clip_fraction)
        self.total_timesteps += length
        self.timesteps.append(self.total_timesteps)

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
        """Return a smoothed reward curve using a causal rolling average."""
        smoothed = []
        for i in range(len(self.rewards)):
            start = max(0, i - window + 1)
            smoothed.append(sum(self.rewards[start : i + 1]) / (i - start + 1))
        return smoothed

    def is_solved(self, threshold: float = 475.0, window: int = 100) -> bool:
        """Check if the environment is solved."""
        if len(self.rewards) < window:
            return False
        return self.moving_average(window) >= threshold

    def summary(self, last_n: int = 100) -> str:
        """Return a formatted summary string."""
        recent = self.rewards[-last_n:] if self.rewards else [0.0]
        return (
            f"Episodes: {len(self.rewards)} | "
            f"Timesteps: {self.total_timesteps} | "
            f"Avg({last_n}): {sum(recent)/len(recent):.1f} | "
            f"Best: {self.best_average(last_n):.1f}"
        )


# ---------------------------------------------------------------------------
# Advantage Computation Helpers
# ---------------------------------------------------------------------------

def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to zero mean and unit variance.

    This is standard practice in PPO and helps stabilize training by
    ensuring the surrogate objective has a consistent scale.

    Args:
        advantages: Tensor of advantage estimates.
        eps: Small constant for numerical stability.

    Returns:
        Normalized advantages with mean ~0 and std ~1.
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def compute_discounted_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute discounted returns G_t from a list of rewards.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: List of rewards from an episode.
        gamma: Discount factor.
        normalize: Whether to normalize returns to zero mean, unit variance.

    Returns:
        Tensor of returns with the same length as rewards.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ---------------------------------------------------------------------------
# Policy Entropy
# ---------------------------------------------------------------------------

def compute_policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of a Categorical policy from logits.

    H[pi] = -sum_a pi(a) * log pi(a)

    Higher entropy means more exploration (more uniform distribution).
    Lower entropy means the policy is more deterministic.

    Args:
        logits: Tensor of shape (..., n_actions) -- raw action logits.

    Returns:
        Scalar entropy value (averaged over batch if applicable).
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


# ---------------------------------------------------------------------------
# Explained Variance
# ---------------------------------------------------------------------------

def compute_explained_variance(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute explained variance: EV = 1 - Var(y_true - y_pred) / Var(y_true).

    Measures how well the value function predicts returns.
    - EV = 1: perfect predictions
    - EV = 0: predicting the mean is equally good
    - EV < 0: worse than predicting the mean

    Args:
        y_pred: Predicted values (V(s)).
        y_true: Actual returns.

    Returns:
        Explained variance as a float.
    """
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0
    return 1.0 - np.var(y_true - y_pred) / var_true


# ---------------------------------------------------------------------------
# Gradient Norm
# ---------------------------------------------------------------------------

def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm across all parameters.

    Useful for monitoring training stability and debugging.

    Args:
        model: PyTorch model (gradients must exist).

    Returns:
        Total L2 norm of all parameter gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ---------------------------------------------------------------------------
# Seed and Device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available torch device."""
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
    """Save a training checkpoint to disk."""
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
    """Load a training checkpoint from disk."""
    return torch.load(filepath, map_location=device, weights_only=False)
