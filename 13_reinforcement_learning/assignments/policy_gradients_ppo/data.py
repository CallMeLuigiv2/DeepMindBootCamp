"""
Rollout Buffers and GAE Computation
=====================================

Provides data infrastructure for policy gradient methods:
- RolloutBuffer: stores rollout data for on-policy methods (pre-written)
- compute_gae: Generalized Advantage Estimation (stubbed)
- compute_returns: Monte Carlo return computation (pre-written)
"""

import numpy as np
import torch
from typing import List, Tuple, Generator, Optional


# ---------------------------------------------------------------------------
# Rollout Buffer (pre-written)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Buffer for storing on-policy rollout data.

    Stores transitions from parallel environments during rollout collection.
    Supports computing returns, advantages, and generating mini-batches
    for PPO updates.

    Args:
        n_steps: Number of steps per environment per rollout.
        n_envs: Number of parallel environments.
        state_dim: Dimensionality of the state space.
        device: Torch device for tensor storage.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        state_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.state_dim = state_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Clear the buffer for a new rollout."""
        self.states = torch.zeros(
            (self.n_steps, self.n_envs, self.state_dim), device=self.device
        )
        self.actions = torch.zeros(
            (self.n_steps, self.n_envs), dtype=torch.long, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.rewards = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.dones = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.values = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.advantages = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.returns = torch.zeros(
            (self.n_steps, self.n_envs), device=self.device
        )
        self.pos = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Add a single timestep of data from all environments.

        Args:
            state: (n_envs, state_dim) -- current states.
            action: (n_envs,) -- actions taken.
            log_prob: (n_envs,) -- log-probabilities under current policy.
            reward: (n_envs,) -- rewards received.
            done: (n_envs,) -- episode termination flags.
            value: (n_envs,) -- value estimates V(s).
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1

    def compute_advantages_and_returns(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns for the stored rollout.

        Args:
            next_value: (n_envs,) -- V(s_T) bootstrap value.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        advantages, returns = compute_gae(
            self.rewards,
            self.values,
            self.dones,
            next_value,
            gamma,
            gae_lambda,
        )
        self.advantages = advantages
        self.returns = returns

    def get_batches(
        self, mini_batch_size: int
    ) -> Generator[dict, None, None]:
        """Generate shuffled mini-batches from the buffer.

        Flattens the (n_steps, n_envs) data into a single batch dimension,
        shuffles, and yields mini-batches.

        Args:
            mini_batch_size: Number of transitions per mini-batch.

        Yields:
            Dictionary with keys: states, actions, log_probs, advantages, returns.
        """
        batch_size = self.n_steps * self.n_envs
        indices = np.random.permutation(batch_size)

        # Flatten all arrays: (n_steps, n_envs, ...) -> (batch_size, ...)
        flat_states = self.states.reshape(batch_size, self.state_dim)
        flat_actions = self.actions.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]

            yield {
                "states": flat_states[batch_indices],
                "actions": flat_actions[batch_indices],
                "log_probs": flat_log_probs[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
            }


# ---------------------------------------------------------------------------
# Monte Carlo Returns (pre-written)
# ---------------------------------------------------------------------------

def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
) -> List[float]:
    """Compute discounted returns G_t for each timestep.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Computed efficiently in reverse order.

    Args:
        rewards: List of rewards from a single episode.
        gamma: Discount factor.

    Returns:
        List of returns G_t, same length as rewards.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation (stubbed)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE smoothly interpolates between TD(0) advantages (lambda=0, low variance,
    high bias) and MC advantages (lambda=1, high variance, low bias).

    Formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

    The returns are computed as: returns = advantages + values.

    Args:
        rewards: Tensor of shape (n_steps, n_envs) -- rewards at each step.
        values: Tensor of shape (n_steps, n_envs) -- V(s_t) estimates.
        dones: Tensor of shape (n_steps, n_envs) -- episode termination flags.
        next_value: Tensor of shape (n_envs,) -- V(s_T) for bootstrapping.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter (0 = TD(0), 1 = MC).

    Returns:
        advantages: Tensor of shape (n_steps, n_envs) -- GAE advantage estimates.
        returns: Tensor of shape (n_steps, n_envs) -- advantage + value targets.
    """
    # YOUR CODE HERE
    # Compute GAE in reverse order (from last step to first):
    #
    # 1. Initialize last_gae = 0
    # 2. For t = T-1, T-2, ..., 0:
    #    a. If t == T-1: next_val = next_value
    #       else: next_val = values[t+1]
    #    b. delta_t = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
    #    c. last_gae = delta_t + gamma * gae_lambda * (1 - dones[t]) * last_gae
    #    d. advantages[t] = last_gae
    # 3. returns = advantages + values
    #
    # Hint: iterate backwards through the time dimension
    raise NotImplementedError("Implement compute_gae")
