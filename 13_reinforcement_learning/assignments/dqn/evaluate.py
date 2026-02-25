"""
DQN Evaluation Script
======================

Loads a trained DQN model and evaluates it on a Gymnasium environment.
Supports rendering episodes, recording videos, and computing evaluation metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --env CartPole-v1
    python evaluate.py --checkpoint checkpoints/best_model.pt --env LunarLander-v2 --render
    python evaluate.py --checkpoint checkpoints/best_model.pt --env LunarLander-v2 --record
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from model import QNetwork, DuelingQNetwork
from data import make_env, make_render_env, make_video_env
from utils import get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes visually")
    parser.add_argument("--record", action="store_true", help="Record episodes to video")
    parser.add_argument("--video_dir", type=str, default="videos", help="Video output directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (pre-written)
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    env_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained Q-network from a checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        env_name: Environment name (to determine state/action dims).
        device: Torch device.

    Returns:
        Loaded Q-network in eval mode.
    """
    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    checkpoint = load_checkpoint(checkpoint_path, device)
    hidden_dim = checkpoint.get("hidden_dim", 128)

    if checkpoint.get("dueling", False):
        model = DuelingQNetwork(state_dim, n_actions, hidden_dim).to(device)
    else:
        model = QNetwork(state_dim, n_actions, hidden_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation functions (stubbed)
# ---------------------------------------------------------------------------

def evaluate_agent(
    model: torch.nn.Module,
    env_name: str,
    n_episodes: int,
    device: torch.device,
    render: bool = False,
) -> dict:
    """Evaluate a trained agent over multiple episodes.

    Args:
        model: Trained Q-network.
        env_name: Gymnasium environment name.
        n_episodes: Number of episodes to evaluate.
        device: Torch device.
        render: Whether to render episodes.

    Returns:
        Dictionary with evaluation metrics:
        - "rewards": list of episode rewards
        - "lengths": list of episode lengths
        - "mean_reward": mean episode reward
        - "std_reward": standard deviation of episode rewards
        - "mean_q_values": mean Q-values across episodes
    """
    # YOUR CODE HERE
    # For each episode:
    #   1. Create environment (with rendering if requested)
    #   2. Reset environment
    #   3. Run greedy policy (always take argmax Q action, no exploration)
    #   4. Record total reward, episode length, and Q-values
    # Return aggregated metrics
    raise NotImplementedError("Implement evaluate_agent")


def compute_q_value_analysis(
    model: torch.nn.Module,
    env_name: str,
    device: torch.device,
    n_states: int = 100,
) -> dict:
    """Analyze Q-value predictions vs actual returns.

    Collects states from the environment, computes predicted Q-values,
    then runs the greedy policy from those states to get actual returns.
    Compares predicted vs actual to measure overestimation.

    Args:
        model: Trained Q-network.
        env_name: Gymnasium environment name.
        device: Torch device.
        n_states: Number of states to analyze.

    Returns:
        Dictionary with:
        - "predicted_q": predicted max Q-values for each state
        - "actual_returns": actual returns achieved from each state
        - "overestimation": predicted - actual for each state
    """
    # YOUR CODE HERE
    # 1. Collect n_states states by running random policy
    # 2. For each state, compute max_a Q(s, a) with the trained model
    # 3. For each state, run the greedy policy to get actual return
    # 4. Compare predicted Q-values vs actual returns
    raise NotImplementedError("Implement compute_q_value_analysis")


def record_agent(
    model: torch.nn.Module,
    env_name: str,
    device: torch.device,
    video_dir: str = "videos",
    n_episodes: int = 5,
) -> None:
    """Record video of the trained agent playing.

    Args:
        model: Trained Q-network.
        env_name: Gymnasium environment name.
        device: Torch device.
        video_dir: Directory to save videos.
        n_episodes: Number of episodes to record.
    """
    # YOUR CODE HERE
    # 1. Create environment with video recording wrapper
    # 2. Run greedy policy for n_episodes
    # 3. Videos are automatically saved by the wrapper
    raise NotImplementedError("Implement record_agent")


# ---------------------------------------------------------------------------
# Main (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()

    print(f"Loading model from: {args.checkpoint}")
    print(f"Environment: {args.env}")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.env, device)

    if args.record:
        print(f"Recording {args.episodes} episodes to {args.video_dir}/")
        record_agent(model, args.env, device, args.video_dir, args.episodes)
    else:
        print(f"Evaluating over {args.episodes} episodes...")
        metrics = evaluate_agent(model, args.env, args.episodes, device, render=args.render)

        print(f"\nResults:")
        print(f"  Mean reward: {metrics['mean_reward']:.1f} +/- {metrics['std_reward']:.1f}")
        print(f"  Min reward:  {min(metrics['rewards']):.1f}")
        print(f"  Max reward:  {max(metrics['rewards']):.1f}")


if __name__ == "__main__":
    main()
