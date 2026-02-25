"""
Policy Gradient Network Architectures
=======================================

Implements neural network architectures for policy gradient methods:
- PolicyNetwork: discrete action policy (outputs Categorical distribution)
- ValueNetwork: state-value estimator V(s)
- ActorCritic: shared-backbone actor-critic for A2C
- PPOActorCritic: separate actor/critic with orthogonal initialization for PPO
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces.

    Maps state vectors to a Categorical distribution over actions.

    Architecture:
        Linear(state_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, n_actions) -> Softmax (via Categorical)

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
        # Build a network that outputs action logits:
        #   Linear(state_dim, hidden_dim) + ReLU
        #   Linear(hidden_dim, hidden_dim) + ReLU
        #   Linear(hidden_dim, n_actions) -- raw logits, no activation
        raise NotImplementedError("Implement PolicyNetwork.__init__")

    def forward(self, state: torch.Tensor) -> Categorical:
        """Compute action distribution for the given state.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Categorical distribution over actions. Sample with .sample(),
            get log-prob with .log_prob(action).
        """
        # YOUR CODE HERE
        # Pass state through the network to get logits.
        # Return Categorical(logits=logits).
        raise NotImplementedError("Implement PolicyNetwork.forward")


class ValueNetwork(nn.Module):
    """State-value network V(s).

    Predicts the expected return from a given state under the current policy.

    Architecture:
        Linear(state_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, hidden_dim) -> ReLU
        Linear(hidden_dim, 1)

    Args:
        state_dim: Dimensionality of the state space.
        hidden_dim: Number of units in hidden layers.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()

        # YOUR CODE HERE
        # Build a network that outputs a single scalar value:
        #   Linear(state_dim, hidden_dim) + ReLU
        #   Linear(hidden_dim, hidden_dim) + ReLU
        #   Linear(hidden_dim, 1) -- scalar output
        raise NotImplementedError("Implement ValueNetwork.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate the value of the given state.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            value: Tensor of shape (batch_size, 1) -- V(s).
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement ValueNetwork.forward")


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network for A2C.

    Uses a shared feature extractor with separate actor (policy) and
    critic (value) heads.

    Architecture:
        Shared:
            Linear(state_dim, hidden_dim) -> ReLU
            Linear(hidden_dim, hidden_dim) -> ReLU
        Actor head:
            Linear(hidden_dim, n_actions) -> Categorical
        Critic head:
            Linear(hidden_dim, 1) -> V(s)

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
        # 1. self.shared: shared feature layers (2 linear + ReLU)
        # 2. self.actor_head: Linear(hidden_dim, n_actions)
        # 3. self.critic_head: Linear(hidden_dim, 1)
        raise NotImplementedError("Implement ActorCritic.__init__")

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """Compute action distribution and state value.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            dist: Categorical distribution over actions.
            value: Tensor of shape (batch_size, 1) -- V(s).
        """
        # YOUR CODE HERE
        # 1. Extract shared features
        # 2. Compute action logits -> Categorical distribution
        # 3. Compute value estimate
        # Return (distribution, value)
        raise NotImplementedError("Implement ActorCritic.forward")


class PPOActorCritic(nn.Module):
    """Separate Actor-Critic network for PPO.

    Uses SEPARATE actor and critic networks (no shared backbone) with
    orthogonal weight initialization and Tanh activations, following
    the standard PPO implementation.

    Actor:
        Linear(state_dim, hidden_dim) -> Tanh
        Linear(hidden_dim, hidden_dim) -> Tanh
        Linear(hidden_dim, n_actions)  [init gain=0.01]

    Critic:
        Linear(state_dim, hidden_dim) -> Tanh
        Linear(hidden_dim, hidden_dim) -> Tanh
        Linear(hidden_dim, 1)          [init gain=1.0]

    Hidden layers use orthogonal init with gain=sqrt(2).

    Args:
        state_dim: Dimensionality of the state space.
        n_actions: Number of discrete actions.
        hidden_dim: Number of units in hidden layers.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # YOUR CODE HERE
        # Build separate actor and critic networks with Tanh activations.
        # Apply orthogonal initialization:
        #   - Hidden layers: gain = sqrt(2)
        #   - Actor output: gain = 0.01
        #   - Critic output: gain = 1.0
        #
        # Hint: use a helper function like:
        #   def init_layer(layer, gain=np.sqrt(2)):
        #       nn.init.orthogonal_(layer.weight, gain=gain)
        #       nn.init.constant_(layer.bias, 0)
        #       return layer
        raise NotImplementedError("Implement PPOActorCritic.__init__")

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log-probability, entropy, and value.

        During rollout collection (action=None): sample a new action.
        During PPO update (action provided): compute log-prob of that action.

        Args:
            state: Tensor of shape (batch_size, state_dim).
            action: Optional tensor of actions to evaluate.

        Returns:
            action: Tensor of shape (batch_size,) -- selected or provided action.
            log_prob: Tensor of shape (batch_size,) -- log pi(a|s).
            entropy: Tensor of shape (batch_size,) -- H[pi(.|s)].
            value: Tensor of shape (batch_size, 1) -- V(s).
        """
        # YOUR CODE HERE
        # 1. Compute action logits from the actor network
        # 2. Create Categorical distribution
        # 3. If action is None, sample; otherwise use the provided action
        # 4. Compute log_prob and entropy
        # 5. Compute value from the critic network
        # Return (action, log_prob, entropy, value)
        raise NotImplementedError("Implement PPOActorCritic.get_action_and_value")

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value only (used for GAE bootstrapping).

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            value: Tensor of shape (batch_size, 1).
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement PPOActorCritic.get_value")
