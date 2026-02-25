"""
DQN Training Script
====================

Trains DQN variants (Naive, Replay-only, Full DQN, Double DQN) on
Gymnasium environments. Supports configuration via YAML and command-line
arguments.

Usage:
    python train.py --env CartPole-v1 --variant dqn
    python train.py --env LunarLander-v2 --variant double_dqn --episodes 1500
    python train.py --env LunarLander-v2 --ablation
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model import QNetwork, DuelingQNetwork
from data import ReplayBuffer, make_env
from utils import (
    EpisodeTracker,
    compute_epsilon,
    set_seed,
    save_checkpoint,
    get_device,
)


# ---------------------------------------------------------------------------
# Configuration loading (pre-written)
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load training configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with command-line arguments."""
    if args.env:
        config["env_name"] = args.env
        # Apply environment-specific overrides if available
        env_key = args.env.lower().replace("-", "").replace("_", "")
        if "lunarlander" in env_key and "lunarlander" in config:
            config.update(config["lunarlander"])
    if args.episodes:
        config["episodes"] = args.episodes
    if args.lr:
        config["learning_rate"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.variant:
        config["variant"] = args.variant
    return config


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN variants")
    parser.add_argument("--env", type=str, default=None, help="Gymnasium environment name")
    parser.add_argument(
        "--variant",
        type=str,
        default="dqn",
        choices=["naive", "replay_only", "dqn", "double_dqn"],
        help="DQN variant to train",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Epsilon-greedy schedule (pre-written)
# ---------------------------------------------------------------------------

def select_action_epsilon_greedy(
    q_network: nn.Module,
    state: np.ndarray,
    epsilon: float,
    n_actions: int,
    device: torch.device,
) -> int:
    """Select an action using epsilon-greedy exploration.

    With probability epsilon, select a random action.
    With probability 1 - epsilon, select the greedy action (argmax Q).

    Args:
        q_network: The Q-network for computing Q-values.
        state: Current state observation.
        epsilon: Current exploration rate.
        n_actions: Number of available actions.
        device: Torch device.

    Returns:
        Selected action index.
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# Training loop (stubbed)
# ---------------------------------------------------------------------------

def train_naive(config: dict, device: torch.device) -> EpisodeTracker:
    """Train naive DQN (no replay, no target network).

    This should demonstrate instability: the agent may learn briefly
    then collapse due to correlated data and moving targets.

    Args:
        config: Training configuration dictionary.
        device: Torch device for computation.

    Returns:
        EpisodeTracker with recorded training metrics.
    """
    env = make_env(config["env_name"], seed=config.get("seed"))
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    tracker = EpisodeTracker()

    # YOUR CODE HERE
    # Training loop:
    # For each episode:
    #   1. Reset environment, get initial state
    #   2. For each step:
    #      a. Select action with epsilon-greedy (use select_action_epsilon_greedy)
    #      b. Take action, observe (next_state, reward, done)
    #      c. Compute TD target: y = r + gamma * max_a' Q(s', a'; theta) (same network!)
    #      d. Compute loss: (y - Q(s, a; theta))^2
    #      e. Backpropagate and update
    #   3. Decay epsilon
    #   4. Track episode reward with tracker.add_episode(reward)
    #
    # WARNING: This approach updates on single transitions using the same
    # network for both target and prediction. It WILL be unstable.
    raise NotImplementedError("Implement train_naive")

    return tracker


def train_with_replay(config: dict, device: torch.device) -> EpisodeTracker:
    """Train DQN with experience replay but no target network.

    Replay breaks temporal correlation but the target still moves with
    each update (same network used for target and prediction).

    Args:
        config: Training configuration dictionary.
        device: Torch device for computation.

    Returns:
        EpisodeTracker with recorded training metrics.
    """
    env = make_env(config["env_name"], seed=config.get("seed"))
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    tracker = EpisodeTracker()

    # YOUR CODE HERE
    # Training loop:
    # For each episode:
    #   1. Reset environment, get initial state
    #   2. For each step:
    #      a. Select action with epsilon-greedy
    #      b. Take action, observe (next_state, reward, done)
    #      c. Store transition in replay buffer
    #      d. If buffer has >= min_buffer_size transitions:
    #         - Sample a mini-batch from the buffer
    #         - Compute TD targets using the SAME network: y = r + gamma * max Q(s'; theta)
    #         - Compute loss and update
    #   3. Decay epsilon
    #   4. Track episode reward
    raise NotImplementedError("Implement train_with_replay")

    return tracker


def train_dqn(config: dict, device: torch.device) -> EpisodeTracker:
    """Train full DQN with experience replay and target network.

    The target network is a frozen copy updated periodically, providing
    a stable target for the TD update.

    Args:
        config: Training configuration dictionary.
        device: Torch device for computation.

    Returns:
        EpisodeTracker with recorded training metrics.
    """
    env = make_env(config["env_name"], seed=config.get("seed"))
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    target_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    tracker = EpisodeTracker()

    total_steps = 0

    # YOUR CODE HERE
    # Training loop:
    # For each episode:
    #   1. Reset environment, get initial state
    #   2. For each step:
    #      a. Select action with epsilon-greedy
    #      b. Take action, observe (next_state, reward, done)
    #      c. Store transition in replay buffer
    #      d. If buffer has >= min_buffer_size transitions:
    #         - Sample a mini-batch
    #         - Compute TD targets using TARGET network: y = r + gamma * max Q(s'; theta^-)
    #         - Compute Huber loss and update ONLINE network
    #         - Update total_steps
    #         - If total_steps % target_update_freq == 0:
    #             Hard update: copy online network weights to target network
    #             OR soft update: theta^- = tau*theta + (1-tau)*theta^-
    #   3. Decay epsilon
    #   4. Track episode reward
    raise NotImplementedError("Implement train_dqn")

    return tracker


def train_double_dqn(config: dict, device: torch.device) -> EpisodeTracker:
    """Train Double DQN to reduce Q-value overestimation.

    The only difference from standard DQN is the target computation:
    - Online network SELECTS the best action: a* = argmax Q(s'; theta)
    - Target network EVALUATES that action: y = r + gamma * Q(s', a*; theta^-)

    This decoupling reduces the upward bias in Q-value estimates.

    Args:
        config: Training configuration dictionary.
        device: Torch device for computation.

    Returns:
        EpisodeTracker with recorded training metrics.
    """
    env = make_env(config["env_name"], seed=config.get("seed"))
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    target_network = QNetwork(state_dim, n_actions, config["hidden_dim"]).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    tracker = EpisodeTracker()

    total_steps = 0

    # YOUR CODE HERE
    # Same as train_dqn but with Double DQN target computation:
    #   1. best_actions = argmax_a' Q(s', a'; theta)    -- online network selects
    #   2. next_q = Q(s', best_actions; theta^-)         -- target network evaluates
    #   3. targets = r + gamma * next_q * (1 - done)
    raise NotImplementedError("Implement train_double_dqn")

    return tracker


# ---------------------------------------------------------------------------
# Ablation study runner (pre-written)
# ---------------------------------------------------------------------------

def run_ablation(config: dict, device: torch.device) -> dict:
    """Run the ablation study on LunarLander-v2.

    Tests the following configurations:
    - Full Double DQN
    - DQN (no Double)
    - No target network (replay only)
    - No experience replay
    - No replay + no target
    - Buffer size = 1000
    - Buffer size = 100000

    Returns:
        Dictionary mapping configuration name to EpisodeTracker.
    """
    ablation_configs = {
        "Full Double DQN": {"variant": "double_dqn"},
        "DQN (no Double)": {"variant": "dqn"},
        "No target network": {"variant": "replay_only"},
        "No replay + no target": {"variant": "naive"},
        "Buffer size = 1000": {"variant": "dqn", "buffer_size": 1000},
        "Buffer size = 100000": {"variant": "dqn", "buffer_size": 100000},
    }

    results = {}
    train_fn_map = {
        "naive": train_naive,
        "replay_only": train_with_replay,
        "dqn": train_dqn,
        "double_dqn": train_double_dqn,
    }

    for name, overrides in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")
        run_config = {**config, **overrides}
        variant = run_config.pop("variant", "dqn")
        train_fn = train_fn_map[variant]
        tracker = train_fn(run_config, device)
        results[name] = tracker

    return results


# ---------------------------------------------------------------------------
# Main entry point (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    device = get_device()
    set_seed(config.get("seed", 42))

    print(f"Environment: {config['env_name']}")
    print(f"Device: {device}")
    print(f"Seed: {config.get('seed', 42)}")

    # Create checkpoint directory
    os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs(config.get("log_dir", "logs"), exist_ok=True)

    if args.ablation:
        results = run_ablation(config, device)
        print("\nAblation study complete. Results saved to logs/")
    else:
        variant = config.get("variant", args.variant)
        train_fn_map = {
            "naive": train_naive,
            "replay_only": train_with_replay,
            "dqn": train_dqn,
            "double_dqn": train_double_dqn,
        }

        print(f"Variant: {variant}")
        train_fn = train_fn_map[variant]
        tracker = train_fn(config, device)

        # Print summary
        print(f"\nTraining complete!")
        print(f"  Final avg reward (last 100): {tracker.moving_average(100):.1f}")
        print(f"  Best avg reward (100-ep window): {tracker.best_average(100):.1f}")


if __name__ == "__main__":
    main()
