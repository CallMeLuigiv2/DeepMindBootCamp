"""
Diffusion Model - Training Script

Trains a DDPM on MNIST with noise prediction objective.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --schedule cosine
    python train.py --config config.yaml --T 500
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml

from data import get_dataloader
from model import UNet, get_noise_schedule, forward_diffusion
from utils import MetricTracker, setup_logger, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--schedule", type=str, default=None, help="Override noise schedule type")
    parser.add_argument("--T", type=int, default=None, help="Override number of timesteps")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: UNet,
    dataloader,
    optimizer: torch.optim.Optimizer,
    alpha_bars: torch.Tensor,
    T: int,
    device: torch.device,
) -> float:
    """Train for one epoch using the DDPM objective.

    For each batch:
        1. Sample random timesteps
        2. Add noise via forward diffusion
        3. Predict the noise with the model
        4. Compute MSE loss between predicted and actual noise
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)

        # 1. Sample random timesteps
        t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)

        # 2. Add noise using the forward process
        x_t, epsilon = forward_diffusion(images, t, alpha_bars.to(device))

        # 3. Predict the noise
        epsilon_pred = model(x_t, t)

        # 4. Compute MSE loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # 5. Backpropagate and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]
    noise_cfg = config["noise_schedule"]
    unet_cfg = config["unet"]

    set_seed(train_cfg["seed"])
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger = setup_logger(config["logging"]["log_dir"])

    # Override from args
    schedule_type = args.schedule or noise_cfg["type"]
    T = args.T or noise_cfg["T"]

    logger.info(f"Device: {device}")
    logger.info(f"Schedule: {schedule_type}, T: {T}")

    # Noise schedule
    betas, alphas, alpha_bars = get_noise_schedule(
        schedule_type, T,
        beta_start=noise_cfg.get("beta_start", 1e-4),
        beta_end=noise_cfg.get("beta_end", 0.02),
    )

    # Data
    dataloader = get_dataloader(
        train_cfg["dataset"], train_cfg["batch_size"],
    )

    # Model
    model = UNet(
        in_channels=unet_cfg["in_channels"],
        base_channels=unet_cfg["base_channels"],
        channel_mults=unet_cfg["channel_mults"],
        time_emb_dim=unet_cfg["time_emb_dim"],
        num_groups=unet_cfg["num_groups"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {param_count:,}")

    # Verify architecture
    test_x = torch.randn(2, 1, 28, 28, device=device)
    test_t = torch.randint(0, T, (2,), device=device)
    test_out = model(test_x, test_t)
    assert test_out.shape == test_x.shape, f"Shape mismatch: {test_out.shape} vs {test_x.shape}"
    logger.info("Architecture verified: output shape matches input shape")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    tracker = MetricTracker()
    best_loss = float("inf")

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, dataloader, optimizer, alpha_bars, T, device)
        elapsed = time.time() - start

        tracker.update("train_loss", train_loss)
        logger.info(f"Epoch {epoch}/{train_cfg['num_epochs']} | Loss: {train_loss:.6f} | {elapsed:.1f}s")

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                os.path.join(config["logging"]["save_dir"], f"best_{schedule_type}_T{T}.pt"),
                model, optimizer, epoch, best_loss,
                extra={"schedule_type": schedule_type, "T": T,
                       "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars},
            )

        if epoch % config["logging"]["save_interval"] == 0:
            save_checkpoint(
                os.path.join(config["logging"]["save_dir"], f"epoch_{epoch}.pt"),
                model, optimizer, epoch, train_loss,
                extra={"schedule_type": schedule_type, "T": T,
                       "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars},
            )

    tracker.plot(
        os.path.join(config["logging"]["log_dir"], f"training_{schedule_type}_T{T}.png"),
        title=f"DDPM Training ({schedule_type}, T={T})",
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
