"""
VAE - Training Script

Trains a VAE (or Conditional VAE) on MNIST with ELBO loss.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --beta 5.0
    python train.py --config config.yaml --conditional
"""

import argparse
import os
import time

import torch
import yaml

from data import get_mnist_dataloaders, flatten_batch
from model import VAE, ConditionalVAE, vae_loss
from utils import MetricTracker, setup_logger, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--beta", type=float, default=None, help="Override beta value")
    parser.add_argument("--conditional", action="store_true", help="Train Conditional VAE")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model, train_loader, optimizer, beta, device, kl_weight=1.0,
):
    """Train for one epoch.

    Args:
        kl_weight: Multiplier for KL annealing (ramps from 0 to 1)

    Returns:
        avg_total_loss, avg_recon_loss, avg_kl_loss
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_samples = 0

    for images, labels in train_loader:
        images = flatten_batch(images).to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if isinstance(model, ConditionalVAE):
            x_recon, mu, log_var = model(images, labels)
        else:
            x_recon, mu, log_var = model(images)

        total, recon, kl = vae_loss(x_recon, images, mu, log_var, beta=beta * kl_weight)
        total.backward()
        optimizer.step()

        total_loss_sum += total.item()
        recon_loss_sum += recon.item()
        kl_loss_sum += kl.item()
        num_samples += batch_size

    return (
        total_loss_sum / num_samples,
        recon_loss_sum / num_samples,
        kl_loss_sum / num_samples,
    )


@torch.no_grad()
def validate(model, test_loader, beta, device):
    """Compute validation loss."""
    model.eval()
    total_loss_sum = 0.0
    num_samples = 0

    for images, labels in test_loader:
        images = flatten_batch(images).to(device)
        labels = labels.to(device)

        if isinstance(model, ConditionalVAE):
            x_recon, mu, log_var = model(images, labels)
        else:
            x_recon, mu, log_var = model(images)

        total, _, _ = vae_loss(x_recon, images, mu, log_var, beta=beta)
        total_loss_sum += total.item()
        num_samples += images.size(0)

    return total_loss_sum / num_samples


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]
    model_cfg = config["model"]

    set_seed(train_cfg["seed"])
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger = setup_logger(config["logging"]["log_dir"])

    beta = args.beta if args.beta is not None else train_cfg["beta"]
    logger.info(f"Device: {device}")
    logger.info(f"Beta: {beta}")
    logger.info(f"Conditional: {args.conditional}")

    # Data
    train_loader, test_loader = get_mnist_dataloaders(train_cfg["batch_size"])

    # Model
    if args.conditional:
        model = ConditionalVAE(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            latent_dim=model_cfg["latent_dim"],
            num_classes=model_cfg["num_classes"],
        ).to(device)
    else:
        model = VAE(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            latent_dim=model_cfg["latent_dim"],
        ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    tracker = MetricTracker()
    best_val_loss = float("inf")

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        # KL annealing
        if train_cfg["kl_annealing"]:
            kl_weight = min(1.0, epoch / train_cfg["kl_annealing_epochs"])
        else:
            kl_weight = 1.0

        start = time.time()
        total, recon, kl = train_one_epoch(
            model, train_loader, optimizer, beta, device, kl_weight
        )
        val_loss = validate(model, test_loader, beta, device)
        elapsed = time.time() - start

        tracker.update("total_loss", total)
        tracker.update("recon_loss", recon)
        tracker.update("kl_loss", kl)
        tracker.update("val_loss", val_loss)

        logger.info(
            f"Epoch {epoch}/{train_cfg['num_epochs']} | "
            f"Total: {total:.2f} | Recon: {recon:.2f} | KL: {kl:.2f} | "
            f"Val: {val_loss:.2f} | KL_w: {kl_weight:.2f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tag = "cvae" if args.conditional else f"vae_beta{beta}"
            save_checkpoint(
                os.path.join(config["logging"]["save_dir"], f"best_{tag}.pt"),
                model, optimizer, epoch, best_val_loss,
            )

    tracker.plot(
        os.path.join(config["logging"]["log_dir"], "training_curves.png"),
        title=f"VAE Training (beta={beta})",
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
