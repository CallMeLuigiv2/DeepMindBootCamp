"""
Policy Gradient Evaluation Script
===================================

Loads a trained policy gradient model and evaluates it on a Gymnasium
environment. Supports rendering and variance analysis.

Usage:
    python evaluate.py --checkpoint checkpoints/ppo_best.pt --env CartPole-v1
    python evaluate.py --checkpoint checkpoints/ppo_best.pt --env LunarLander-v2 --render
"""

import argparse

import gymnasium as gym
import numpy as np
import torch

from model import PolicyNetwork, ActorCritic, PPOActorCritic
from utils import get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policy gradient agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions (no sampling)")
    parser.add_argument(
        "--method",
        type=str,
        default="ppo",
        choices=["reinforce", "reinforce_baseline", "a2c", "ppo"],
        help="Method used for training (determines model class)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (pre-written)
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    env_name: str,
    method: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained policy model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        env_name: Environment name for determining dimensions.
        method: Training method (determines model architecture).
        device: Torch device.

    Returns:
        Loaded model in eval mode.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    checkpoint = load_checkpoint(checkpoint_path, device)
    hidden_dim = checkpoint.get("hidden_dim", 128)

    if method == "ppo":
        model = PPOActorCritic(state_dim, n_actions, hidden_dim).to(device)
    elif method in ("a2c",):
        model = ActorCritic(state_dim, n_actions, hidden_dim).to(device)
    else:
        model = PolicyNetwork(state_dim, n_actions, hidden_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation (stubbed)
# ---------------------------------------------------------------------------

def evaluate_agent(
    model: torch.nn.Module,
    env_name: str,
    method: str,
    n_episodes: int,
    device: torch.device,
    render: bool = False,
    deterministic: bool = False,
) -> dict:
    """Evaluate a trained policy over multiple episodes.

    Args:
        model: Trained policy model.
        env_name: Gymnasium environment name.
        method: Training method (for action selection).
        n_episodes: Number of evaluation episodes.
        device: Torch device.
        render: Whether to render episodes.
        deterministic: If True, take argmax action instead of sampling.

    Returns:
        Dictionary with:
        - "rewards": list of episode rewards
        - "lengths": list of episode lengths
        - "mean_reward": mean reward
        - "std_reward": std of rewards
    """
    # YOUR CODE HERE
    # For each episode:
    #   1. Create environment (with render_mode if render=True)
    #   2. Reset environment
    #   3. Run policy:
    #      - If deterministic: take argmax action
    #      - Otherwise: sample from the policy distribution
    #   4. Record total reward and episode length
    # Return aggregated metrics
    raise NotImplementedError("Implement evaluate_agent")


def compute_gradient_variance(
    model: torch.nn.Module,
    env_name: str,
    method: str,
    device: torch.device,
    n_episodes: int = 20,
) -> dict:
    """Estimate policy gradient variance across multiple episodes.

    Collects episodes and computes the policy gradient for each one.
    Reports the mean and standard deviation of gradient norms.

    Args:
        model: Trained policy model.
        env_name: Environment name.
        method: Training method.
        device: Torch device.
        n_episodes: Number of episodes to collect.

    Returns:
        Dictionary with:
        - "grad_norms": list of gradient norms
        - "mean_grad_norm": mean gradient norm
        - "std_grad_norm": std of gradient norms
    """
    # YOUR CODE HERE
    # For each episode:
    #   1. Collect episode with current policy
    #   2. Compute REINFORCE-style gradient (log_prob * return)
    #   3. Compute gradient norm
    # Report statistics across episodes
    raise NotImplementedError("Implement compute_gradient_variance")


# ---------------------------------------------------------------------------
# Main (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()

    print(f"Loading model from: {args.checkpoint}")
    print(f"Environment: {args.env}")
    print(f"Method: {args.method}")

    model = load_model(args.checkpoint, args.env, args.method, device)

    print(f"Evaluating over {args.episodes} episodes...")
    metrics = evaluate_agent(
        model, args.env, args.method, args.episodes, device,
        render=args.render, deterministic=args.deterministic,
    )

    print(f"\nResults:")
    print(f"  Mean reward: {metrics['mean_reward']:.1f} +/- {metrics['std_reward']:.1f}")
    print(f"  Min reward:  {min(metrics['rewards']):.1f}")
    print(f"  Max reward:  {max(metrics['rewards']):.1f}")


if __name__ == "__main__":
    main()
