"""
Replay Buffers and Environment Wrappers
========================================

Provides data infrastructure for DQN training:
- ReplayBuffer: uniform random sampling replay buffer (pre-written)
- PrioritizedReplayBuffer: proportional prioritization by TD error (stubbed)
- Environment wrapper utilities
"""

import numpy as np
import gymnasium as gym
from collections import deque
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Replay Buffer (pre-written)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform random sampling.

    Stores transitions (s, a, r, s', done) in a circular buffer. When full,
    the oldest transitions are overwritten. Sampling is uniform at random.

    This breaks temporal correlation between consecutive transitions, which is
    critical for stable DQN training.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones),
            each as a numpy array with batch_size as the first dimension.
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer (stubbed -- stretch goal)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to |TD error|^alpha.
    Uses importance sampling weights to correct for the non-uniform sampling bias.

    Reference: Schaul et al., "Prioritized Experience Replay" (2016).

    Args:
        capacity: Maximum number of transitions to store.
        alpha: Exponent for prioritization (0 = uniform, 1 = fully prioritized).
        beta_start: Initial importance sampling exponent (annealed to 1.0).
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start

        # YOUR CODE HERE
        # Initialize storage for transitions, priorities, and a position pointer.
        # You will need:
        #   - A storage array or list for transitions
        #   - A priorities array (numpy) initialized to a small epsilon
        #   - A position counter and current size tracker
        raise NotImplementedError("Implement PrioritizedReplayBuffer.__init__")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with maximum priority.

        New transitions get the maximum priority in the buffer so they
        are sampled at least once.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement PrioritizedReplayBuffer.push")

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch with prioritized probabilities.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
            - indices: buffer indices of sampled transitions (for updating priorities)
            - weights: importance sampling weights (normalized to max weight = 1)
        """
        # YOUR CODE HERE
        # 1. Compute sampling probabilities: P(i) = p_i^alpha / sum(p_j^alpha)
        # 2. Sample batch_size indices according to P
        # 3. Compute importance sampling weights: w_i = (N * P(i))^{-beta}
        # 4. Normalize weights by dividing by max(w)
        raise NotImplementedError("Implement PrioritizedReplayBuffer.sample")

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Buffer indices of the transitions to update.
            td_errors: Absolute TD errors for the corresponding transitions.
        """
        # YOUR CODE HERE
        # Set priority = |td_error| + small_epsilon to avoid zero priority
        raise NotImplementedError("Implement PrioritizedReplayBuffer.update_priorities")

    def __len__(self) -> int:
        # YOUR CODE HERE
        raise NotImplementedError("Implement PrioritizedReplayBuffer.__len__")


# ---------------------------------------------------------------------------
# Environment Wrappers (pre-written)
# ---------------------------------------------------------------------------

def make_env(env_name: str, seed: Optional[int] = None) -> gym.Env:
    """Create and configure a Gymnasium environment.

    Args:
        env_name: Name of the Gymnasium environment (e.g., "CartPole-v1").
        seed: Random seed for reproducibility.

    Returns:
        Configured Gymnasium environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_render_env(env_name: str) -> gym.Env:
    """Create an environment with human rendering enabled.

    Args:
        env_name: Name of the Gymnasium environment.

    Returns:
        Environment configured for visual rendering.
    """
    return gym.make(env_name, render_mode="human")


def make_video_env(env_name: str, video_dir: str = "videos") -> gym.Env:
    """Create an environment that records episodes to video.

    Args:
        env_name: Name of the Gymnasium environment.
        video_dir: Directory to save recorded videos.

    Returns:
        Environment wrapped with video recording.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_dir)
    return env
