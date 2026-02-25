"""
Policy Gradient Training Script
================================

Trains policy gradient methods (REINFORCE, REINFORCE+Baseline, A2C, PPO)
on Gymnasium environments.

Usage:
    python train.py --method reinforce --env CartPole-v1
    python train.py --method ppo --env CartPole-v1 --total_timesteps 100000
    python train.py --method ppo --env LunarLander-v2 --total_timesteps 500000
    python train.py --compare --seeds 5
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model import PolicyNetwork, ValueNetwork, ActorCritic, PPOActorCritic
from data import RolloutBuffer, compute_returns, compute_gae
from utils import (
    EpisodeTracker,
    set_seed,
    get_device,
    save_checkpoint,
    normalize_advantages,
    compute_policy_entropy,
    compute_explained_variance,
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


# ---------------------------------------------------------------------------
# Argument parsing (pre-written)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy gradient methods")
    parser.add_argument(
        "--method",
        type=str,
        default="ppo",
        choices=["reinforce", "reinforce_baseline", "a2c", "ppo"],
        help="Training method",
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (REINFORCE/A2C)")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps (PPO)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Run method comparison")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for comparison")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parallel environment setup (pre-written)
# ---------------------------------------------------------------------------

def make_envs(env_name: str, n_envs: int, seed: int = 42) -> gym.vector.VectorEnv:
    """Create vectorized parallel environments for PPO.

    Args:
        env_name: Gymnasium environment name.
        n_envs: Number of parallel environments.
        seed: Base random seed.

    Returns:
        SyncVectorEnv with n_envs parallel environments.
    """
    def make_env(env_seed):
        def _make():
            env = gym.make(env_name)
            env.reset(seed=env_seed)
            return env
        return _make

    envs = gym.vector.SyncVectorEnv(
        [make_env(seed + i) for i in range(n_envs)]
    )
    return envs


# ---------------------------------------------------------------------------
# REINFORCE (stubbed)
# ---------------------------------------------------------------------------

def train_reinforce(config: dict, device: torch.device) -> EpisodeTracker:
    """Train using the REINFORCE algorithm.

    The simplest policy gradient method. Collects complete episodes, computes
    Monte Carlo returns G_t, and updates the policy with:
        loss = -sum_t log(pi(a_t|s_t)) * G_t

    Args:
        config: Training configuration.
        device: Torch device.

    Returns:
        EpisodeTracker with training metrics.
    """
    env = gym.make(config["env_name"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    rc = config.get("reinforce", {})
    policy = PolicyNetwork(state_dim, n_actions, config.get("hidden_dim", 128)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=rc.get("learning_rate", 1e-3))
    gamma = rc.get("gamma", 0.99)
    n_episodes = config.get("episodes") or rc.get("episodes", 1000)

    tracker = EpisodeTracker()

    # YOUR CODE HERE
    # For each episode:
    #   1. Reset environment
    #   2. Collect full episode: store (log_probs, rewards) at each step
    #      - Get distribution from policy(state)
    #      - Sample action, record log_prob
    #   3. Compute returns G_t using compute_returns(rewards, gamma)
    #   4. Compute policy loss: L = -sum(log_prob * G_t)
    #      (optionally normalize returns for stability)
    #   5. Backpropagate and update
    #   6. Track episode reward
    raise NotImplementedError("Implement train_reinforce")

    return tracker


# ---------------------------------------------------------------------------
# REINFORCE with Baseline (stubbed)
# ---------------------------------------------------------------------------

def train_reinforce_baseline(config: dict, device: torch.device) -> EpisodeTracker:
    """Train using REINFORCE with a learned value baseline.

    Subtracts V(s_t) from the return to reduce gradient variance:
        advantage = G_t - V(s_t)
        policy_loss = -sum_t log(pi(a_t|s_t)) * advantage
        value_loss = sum_t (G_t - V(s_t))^2

    The baseline does NOT change the expected gradient. It only reduces
    variance, making learning faster and more stable.

    Args:
        config: Training configuration.
        device: Torch device.

    Returns:
        EpisodeTracker with training metrics.
    """
    env = gym.make(config["env_name"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    rc = config.get("reinforce_baseline", {})
    policy = PolicyNetwork(state_dim, n_actions, config.get("hidden_dim", 128)).to(device)
    value_net = ValueNetwork(state_dim, config.get("hidden_dim", 128)).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=rc.get("lr_policy", 1e-3))
    value_optimizer = optim.Adam(value_net.parameters(), lr=rc.get("lr_value", 1e-3))
    gamma = rc.get("gamma", 0.99)
    n_episodes = config.get("episodes") or rc.get("episodes", 1000)

    tracker = EpisodeTracker()

    # YOUR CODE HERE
    # For each episode:
    #   1. Reset environment
    #   2. Collect full episode: store (states, log_probs, rewards) at each step
    #   3. Compute returns G_t
    #   4. Compute baselines: V(s_t) for each state
    #   5. Compute advantages: A_t = G_t - V(s_t)
    #   6. Policy loss: -sum(log_prob * A_t.detach())
    #   7. Value loss: MSE(V(s_t), G_t)
    #   8. Update both networks
    #   9. Track episode reward
    raise NotImplementedError("Implement train_reinforce_baseline")

    return tracker


# ---------------------------------------------------------------------------
# A2C (stubbed)
# ---------------------------------------------------------------------------

def train_a2c(config: dict, device: torch.device) -> EpisodeTracker:
    """Train using Advantage Actor-Critic (A2C).

    Uses one-step TD advantages instead of MC returns, and updates at
    every step (not just episode end).

    Advantage: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    Actor loss: -mean(log pi(a_t|s_t) * A_t.detach())
    Critic loss: mean(A_t^2)
    Entropy bonus: -mean(H[pi(.|s_t)])

    Total loss = actor_loss + value_coeff * critic_loss + entropy_coeff * entropy_loss

    Args:
        config: Training configuration.
        device: Torch device.

    Returns:
        EpisodeTracker with training metrics.
    """
    env = gym.make(config["env_name"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    ac = config.get("a2c", {})
    model = ActorCritic(state_dim, n_actions, config.get("hidden_dim", 128)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ac.get("learning_rate", 3e-4))
    gamma = ac.get("gamma", 0.99)
    entropy_coeff = ac.get("entropy_coeff", 0.01)
    value_coeff = ac.get("value_loss_coeff", 0.5)
    n_episodes = config.get("episodes") or ac.get("episodes", 1000)

    tracker = EpisodeTracker()

    # YOUR CODE HERE
    # For each episode:
    #   1. Reset environment
    #   2. For each step:
    #      a. Get (distribution, value) from model(state)
    #      b. Sample action, record log_prob
    #      c. Take action, get (next_state, reward, done)
    #      d. Compute next_value = model(next_state)[1] if not done, else 0
    #      e. Advantage = reward + gamma * next_value - value
    #      f. Actor loss = -log_prob * advantage.detach()
    #      g. Critic loss = advantage^2
    #      h. Entropy = distribution.entropy()
    #      i. Total loss = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy
    #      j. Backpropagate and update
    #   3. Track episode reward
    #
    # CRITICAL: detach advantage before using in actor loss!
    raise NotImplementedError("Implement train_a2c")

    return tracker


# ---------------------------------------------------------------------------
# PPO (stubbed)
# ---------------------------------------------------------------------------

def train_ppo(config: dict, device: torch.device) -> EpisodeTracker:
    """Train using Proximal Policy Optimization (PPO).

    Full PPO implementation with:
    - Parallel environments for data collection
    - GAE advantage estimation
    - Clipped surrogate objective
    - Multiple epochs per rollout
    - Mini-batch updates
    - Entropy bonus and gradient clipping
    - Optional learning rate annealing

    Args:
        config: Training configuration.
        device: Torch device.

    Returns:
        EpisodeTracker with training metrics.
    """
    pc = config.get("ppo", {})
    env_name = config.get("env_name", "CartPole-v1")

    # Use LunarLander-specific config if applicable
    if "lunarlander" in env_name.lower().replace("-", ""):
        lunar_cfg = config.get("lunarlander_ppo", {})
        pc = {**pc, **lunar_cfg}

    n_envs = pc.get("n_envs", 8)
    n_steps = pc.get("n_steps", 128)
    n_epochs = pc.get("n_epochs", 4)
    mini_batch_size = pc.get("mini_batch_size", 32)
    total_timesteps = config.get("total_timesteps") or pc.get("total_timesteps", 100000)
    lr = config.get("lr") or pc.get("learning_rate", 2.5e-4)
    gamma = pc.get("gamma", 0.99)
    gae_lambda = pc.get("gae_lambda", 0.95)
    clip_ratio = pc.get("clip_ratio", 0.2)
    entropy_coeff = pc.get("entropy_coeff", 0.01)
    value_coeff = pc.get("value_loss_coeff", 0.5)
    max_grad_norm = pc.get("max_grad_norm", 0.5)
    anneal_lr = pc.get("anneal_lr", True)
    hidden_dim = pc.get("hidden_dim", 64)

    # Create parallel environments
    seed = config.get("seed", 42)
    envs = make_envs(env_name, n_envs, seed)
    state_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    # Create model and optimizer
    model = PPOActorCritic(state_dim, n_actions, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Create rollout buffer
    buffer = RolloutBuffer(n_steps, n_envs, state_dim, device)

    tracker = EpisodeTracker()
    n_updates = total_timesteps // (n_steps * n_envs)

    print(f"PPO Training:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Batch size: {n_steps * n_envs}")
    print(f"  Mini-batch size: {mini_batch_size}")
    print(f"  Number of updates: {n_updates}")

    # YOUR CODE HERE
    # Main training loop:
    # For each update (1 to n_updates):
    #
    #   1. LEARNING RATE ANNEALING (if anneal_lr):
    #      lr_now = lr * (1 - update / n_updates)
    #      update optimizer learning rate
    #
    #   2. COLLECT ROLLOUT:
    #      For each step (0 to n_steps-1):
    #        a. Get (action, log_prob, entropy, value) from model
    #        b. Take actions in envs: next_obs, rewards, terminateds, truncateds, infos
    #        c. Store in buffer: buffer.add(obs, action, log_prob, reward, done, value)
    #        d. Track completed episodes (check infos for final rewards)
    #
    #   3. COMPUTE ADVANTAGES:
    #      a. Get next_value = model.get_value(next_obs)
    #      b. buffer.compute_advantages_and_returns(next_value, gamma, gae_lambda)
    #
    #   4. PPO UPDATE:
    #      For each epoch (0 to n_epochs-1):
    #        For each mini-batch from buffer.get_batches(mini_batch_size):
    #          a. Get new (_, new_log_prob, new_entropy, new_value) for stored (state, action)
    #          b. Compute ratio: r = exp(new_log_prob - old_log_prob)
    #          c. Normalize advantages (zero mean, unit variance)
    #          d. Clipped surrogate: L_clip = min(r*A, clip(r,1-eps,1+eps)*A)
    #          e. Value loss: MSE(new_value, returns)
    #          f. Entropy bonus: mean entropy
    #          g. Total loss = -L_clip + value_coeff * value_loss - entropy_coeff * entropy
    #          h. Backprop with gradient clipping (max_grad_norm)
    #
    #   5. Reset buffer for next rollout
    #
    raise NotImplementedError("Implement train_ppo")

    envs.close()
    return tracker


# ---------------------------------------------------------------------------
# Method comparison runner (pre-written)
# ---------------------------------------------------------------------------

def run_comparison(config: dict, device: torch.device, n_seeds: int = 5) -> dict:
    """Run all methods with multiple seeds for fair comparison.

    Args:
        config: Training configuration.
        device: Torch device.
        n_seeds: Number of random seeds.

    Returns:
        Dictionary mapping method name to list of EpisodeTrackers.
    """
    methods = {
        "REINFORCE": train_reinforce,
        "REINFORCE + Baseline": train_reinforce_baseline,
        "A2C": train_a2c,
        "PPO": train_ppo,
    }

    results = {}
    for name, train_fn in methods.items():
        results[name] = []
        for seed in range(n_seeds):
            print(f"\n{'='*60}")
            print(f"{name} -- Seed {seed}")
            print(f"{'='*60}")
            run_config = {**config, "seed": seed}
            set_seed(seed)
            tracker = train_fn(run_config, device)
            results[name].append(tracker)

    return results


# ---------------------------------------------------------------------------
# Main entry point (pre-written)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.env:
        config["env_name"] = args.env
    if args.episodes:
        config["episodes"] = args.episodes
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.lr:
        config["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed

    device = get_device()
    set_seed(config.get("seed", 42))

    print(f"Environment: {config['env_name']}")
    print(f"Device: {device}")

    os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs(config.get("log_dir", "logs"), exist_ok=True)

    if args.compare:
        results = run_comparison(config, device, n_seeds=args.seeds)
        print("\nComparison complete. Results saved to logs/")
    else:
        train_fn_map = {
            "reinforce": train_reinforce,
            "reinforce_baseline": train_reinforce_baseline,
            "a2c": train_a2c,
            "ppo": train_ppo,
        }

        print(f"Method: {args.method}")
        train_fn = train_fn_map[args.method]
        tracker = train_fn(config, device)

        print(f"\nTraining complete!")
        print(f"  Final avg reward (last 100): {tracker.moving_average(100):.1f}")
        print(f"  Best avg reward (100-ep window): {tracker.best_average(100):.1f}")


if __name__ == "__main__":
    main()
