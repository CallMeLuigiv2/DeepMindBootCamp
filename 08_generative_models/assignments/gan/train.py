"""
GAN - Training Script

Trains basic GAN, DCGAN, or WGAN-GP on the specified dataset.

Usage:
    python train.py --config config.yaml --model basic --dataset mnist
    python train.py --config config.yaml --model dcgan --dataset cifar10
    python train.py --config config.yaml --model wgan_gp --dataset cifar10
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml

from data import get_dataloader
from model import (
    BasicGenerator, BasicDiscriminator,
    DCGANGenerator, DCGANDiscriminator,
    gradient_penalty,
)
from utils import (
    MetricTracker, setup_logger, save_checkpoint, set_seed,
    weights_init, save_sample_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, choices=["basic", "dcgan", "wgan_gp"], required=True)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "celeba"])
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_basic_gan(config: dict, device: torch.device, logger):
    """Train basic MLP GAN on MNIST."""
    cfg = config["basic"]

    dataloader = get_dataloader("mnist", cfg["batch_size"])

    generator = BasicGenerator(cfg["latent_dim"]).to(device)
    discriminator = BasicDiscriminator().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr_g"], betas=tuple(cfg["betas"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr_d"], betas=tuple(cfg["betas"]))
    criterion = nn.BCELoss()

    tracker = MetricTracker()

    for epoch in range(1, cfg["num_epochs"] + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        d_real_epoch = 0.0
        d_fake_epoch = 0.0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # === Train Discriminator ===
            # YOUR CODE HERE
            # 1. Forward on real images, compute BCE loss with real_labels
            # 2. Generate fake images, forward on fakes (detached), compute loss with fake_labels
            # 3. Total D loss = loss_real + loss_fake
            # 4. Backprop and update D
            raise NotImplementedError("Implement discriminator training step")

            # === Train Generator ===
            # YOUR CODE HERE
            # 1. Generate fake images
            # 2. Forward through D (no detach)
            # 3. G loss = BCE(D(fake), real_labels)  -- G wants D to think fakes are real
            # 4. Backprop and update G
            raise NotImplementedError("Implement generator training step")

        avg_g = g_loss_epoch / len(dataloader)
        avg_d = d_loss_epoch / len(dataloader)
        tracker.update("g_loss", avg_g)
        tracker.update("d_loss", avg_d)

        logger.info(f"Epoch {epoch}/{cfg['num_epochs']} | G: {avg_g:.4f} | D: {avg_d:.4f}")

        if epoch % config["logging"]["sample_interval"] == 0:
            save_sample_grid(
                generator, cfg["latent_dim"], device, 64,
                os.path.join(config["logging"]["log_dir"], f"samples_epoch_{epoch:03d}.png"),
            )

    tracker.plot(os.path.join(config["logging"]["log_dir"], "basic_gan_curves.png"))


def train_dcgan(config: dict, dataset_name: str, device: torch.device, logger):
    """Train DCGAN on CIFAR-10 or CelebA."""
    cfg = config["dcgan"]
    nc = 1 if dataset_name == "mnist" else 3

    dataloader = get_dataloader(dataset_name, cfg["batch_size"], cfg["image_size"])

    generator = DCGANGenerator(cfg["latent_dim"], cfg["ngf"], nc).to(device)
    discriminator = DCGANDiscriminator(nc, cfg["ndf"], use_sigmoid=True).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    criterion = nn.BCELoss()

    tracker = MetricTracker()

    for epoch in range(1, cfg["num_epochs"] + 1):
        # YOUR CODE HERE - same pattern as basic GAN but with DCGAN models
        raise NotImplementedError("Implement DCGAN training loop")

    tracker.plot(os.path.join(config["logging"]["log_dir"], "dcgan_curves.png"))


def train_wgan_gp(config: dict, dataset_name: str, device: torch.device, logger):
    """Train WGAN-GP on CIFAR-10 or CelebA."""
    cfg = config["wgan_gp"]
    nc = 1 if dataset_name == "mnist" else 3

    dataloader = get_dataloader(dataset_name, cfg["batch_size"], config["dcgan"]["image_size"])

    generator = DCGANGenerator(cfg["latent_dim"], cfg["ngf"], nc).to(device)
    critic = DCGANDiscriminator(nc, cfg["ndf"], use_sigmoid=False).to(device)

    generator.apply(weights_init)
    critic.apply(weights_init)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))

    tracker = MetricTracker()

    for epoch in range(1, cfg["num_epochs"] + 1):
        # YOUR CODE HERE
        # For each batch:
        #   1. Train critic for n_critic steps:
        #      - critic_loss = critic(fake).mean() - critic(real).mean() + lambda_gp * GP
        #   2. Train generator for 1 step:
        #      - gen_loss = -critic(fake).mean()
        raise NotImplementedError("Implement WGAN-GP training loop")

    tracker.plot(os.path.join(config["logging"]["log_dir"], "wgan_gp_curves.png"))


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    logger = setup_logger(config["logging"]["log_dir"])

    logger.info(f"Device: {device}, Model: {args.model}, Dataset: {args.dataset}")

    if args.model == "basic":
        train_basic_gan(config, device, logger)
    elif args.model == "dcgan":
        train_dcgan(config, args.dataset, device, logger)
    elif args.model == "wgan_gp":
        train_wgan_gp(config, args.dataset, device, logger)


if __name__ == "__main__":
    main()
