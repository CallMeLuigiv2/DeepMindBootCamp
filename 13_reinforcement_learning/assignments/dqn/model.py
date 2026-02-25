"""
DQN Network Architectures
==========================

Implements neural network architectures for Deep Q-Learning:
- QNetwork: standard fully-connected Q-network
- DuelingQNetwork: dueling architecture with separate value and advantage streams

Both networks take a state vector as input and output Q-values for all actions.
"""

import torch
import torch.nn as nn
from typing import Optional


class QNetwork(nn.Module):
    """Standard Q-Network for DQN.

    Maps state vectors to Q-values for each action using a fully-connected
    network with ReLU activations.

    Architecture:
        Linear(state_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, n_actions)

    Args:
        state_dim: Dimensionality of the state space.
        n_actions: Number of discrete actions.
        hidden_dim: Number of units in hidden layers.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # YOUR CODE HERE
        # Build the network as described in the docstring.
        # Use nn.Sequential or define layers individually.
        # Layers:
        #   1. Linear(state_dim, hidden_dim) + ReLU
        #   2. Linear(hidden_dim, hidden_dim) + ReLU
        #   3. Linear(hidden_dim, n_actions) -- no activation (raw Q-values)
        raise NotImplementedError("Implement QNetwork.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given a state.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            q_values: Tensor of shape (batch_size, n_actions) -- Q(s, a)
                      for each action a.
        """
        # YOUR CODE HERE
        # Pass the state through the network and return Q-values.
        raise NotImplementedError("Implement QNetwork.forward")


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network (stretch goal).

    Separates the Q-value into a state-value V(s) and an advantage A(s, a):
        Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

    Subtracting the mean advantage ensures identifiability: the value stream
    truly captures V(s) rather than an arbitrary offset.

    Architecture:
        Shared:
            Linear(state_dim, hidden_dim) -> ReLU
            Linear(hidden_dim, hidden_dim) -> ReLU
        Value stream:
            Linear(hidden_dim, 1) -> V(s)
        Advantage stream:
            Linear(hidden_dim, n_actions) -> A(s, a)

    Args:
        state_dim: Dimensionality of the state space.
        n_actions: Number of discrete actions.
        hidden_dim: Number of units in hidden layers.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # YOUR CODE HERE
        # Build three components:
        # 1. Shared feature extractor (2 linear layers with ReLU)
        # 2. Value stream: Linear(hidden_dim, 1)
        # 3. Advantage stream: Linear(hidden_dim, n_actions)
        raise NotImplementedError("Implement DuelingQNetwork.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values using the dueling architecture.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            q_values: Tensor of shape (batch_size, n_actions).
                      Computed as V(s) + A(s,a) - mean(A(s,:))
        """
        # YOUR CODE HERE
        # 1. Extract shared features.
        # 2. Compute V(s) from the value stream.
        # 3. Compute A(s, a) from the advantage stream.
        # 4. Combine: Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]
        raise NotImplementedError("Implement DuelingQNetwork.forward")
