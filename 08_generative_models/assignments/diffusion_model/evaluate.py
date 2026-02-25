"""
Diffusion Model - Evaluation Script

Generates samples via DDPM reverse process, visualizes denoising trajectories,
and analyzes noise predictions.

Usage:
    python evaluate.py --checkpoint checkpoints/best_linear_T1000.pt --generate
    python evaluate.py --checkpoint checkpoints/best_linear_T1000.pt --trajectories
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import denormalize
from model import UNet
from utils import set_seed, plot_sample_grid, plot_denoising_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDPM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generate", action="store_true", help="Generate sample grid")
    parser.add_argument("--trajectories", action="store_true", help="Show denoising trajectories")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


@torch.no_grad()
def sample_ddpm(
    model: UNet,
    shape: tuple,
    T: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    save_trajectory: bool = False,
    trajectory_steps: list = None,
) -> tuple:
    """Generate samples by iterative denoising from pure noise.

    Args:
        model: Trained noise prediction U-Net
        shape: (B, C, H, W) - Shape of samples to generate
        T: Number of diffusion steps
        betas, alphas, alpha_bars: Noise schedule tensors
        device: torch device
        save_trajectory: Whether to save intermediate states
        trajectory_steps: Which timesteps to save

    Returns:
        x_0: (B, C, H, W) - Generated samples
        trajectory: List of (timestep, x_t) tuples (if save_trajectory=True)
    """
    # YOUR CODE HERE
    # 1. Start from pure noise: x = randn(shape)
    # 2. For t in reversed(range(T)):
    #    a. Predict noise: epsilon_pred = model(x, t)
    #    b. Compute mean: (1/sqrt(alpha_t)) * (x - beta_t/sqrt(1-alpha_bar_t) * epsilon_pred)
    #    c. Add noise if t > 0: x = mean + sqrt(beta_t) * randn
    #    d. If t == 0: x = mean (no noise at final step)
    #    e. Optionally save x at trajectory_steps
    # 3. Return final x_0 and trajectory
    raise NotImplementedError("Implement sample_ddpm")


def evaluate_noise_predictions(model, test_loader, alpha_bars, T, device):
    """Analyze noise prediction accuracy at different timesteps.

    For several timesteps, show:
    - The actual noise added
    - The predicted noise
    - The residual (actual - predicted)
    """
    # YOUR CODE HERE
    # 1. Take a batch of test images
    # 2. For t in [10, 100, 500, 900]:
    #    a. Forward diffuse to get x_t and epsilon
    #    b. Predict epsilon_pred
    #    c. Display: actual noise, predicted noise, residual
    raise NotImplementedError("Implement evaluate_noise_predictions")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    T = checkpoint["extra"]["T"]
    betas = checkpoint["extra"]["betas"]
    alphas = checkpoint["extra"]["alphas"]
    alpha_bars = checkpoint["extra"]["alpha_bars"]

    # TODO: Reconstruct model and load state_dict
    # model = UNet(...)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)
    # model.eval()

    if args.generate:
        print(f"Generating {args.num_samples} samples with T={T}...")
        start = time.time()
        # samples, _ = sample_ddpm(model, (args.num_samples, 1, 28, 28), T, betas, alphas, alpha_bars, device)
        # elapsed = time.time() - start
        # print(f"Sampling took {elapsed:.1f}s")
        # samples = denormalize(samples)
        # plot_sample_grid(samples, save_path=os.path.join(args.output_dir, "generated_samples.png"))
        print("TODO: Load model and generate samples")

    if args.trajectories:
        print("Generating denoising trajectories...")
        traj_steps = config["sampling"]["trajectory_steps"]
        # samples, trajectory = sample_ddpm(
        #     model, (5, 1, 28, 28), T, betas, alphas, alpha_bars, device,
        #     save_trajectory=True, trajectory_steps=traj_steps,
        # )
        # for i in range(5):
        #     plot_denoising_trajectory(
        #         trajectory, i, save_path=os.path.join(args.output_dir, f"trajectory_{i}.png")
        #     )
        print("TODO: Load model and generate trajectories")


if __name__ == "__main__":
    main()
