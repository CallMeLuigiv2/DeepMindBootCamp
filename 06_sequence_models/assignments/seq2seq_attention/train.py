"""
Seq2Seq with Attention - Training Script

Trains the encoder-decoder model with or without attention on the
date format conversion task.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --use-attention
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from data import create_dataloaders, PAD_IDX
from model import Encoder, Decoder, BahdanauAttention, Seq2Seq
from utils import (
    MetricTracker,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Seq2Seq with Attention")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--use-attention", action="store_true", help="Use Bahdanau attention"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(
    src_vocab_size: int,
    trg_vocab_size: int,
    config: dict,
    use_attention: bool,
    device: torch.device,
) -> Seq2Seq:
    """Build the Seq2Seq model from config."""
    model_cfg = config["model"]

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )

    attention = None
    if use_attention:
        attention = BahdanauAttention(
            encoder_hidden_size=model_cfg["hidden_size"],
            decoder_hidden_size=model_cfg["hidden_size"],
            attention_size=model_cfg["attention_size"],
        )

    decoder = Decoder(
        vocab_size=trg_vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        encoder_hidden_size=model_cfg["hidden_size"],
        attention=attention,
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


def train_one_epoch(
    model: Seq2Seq,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    teacher_forcing_ratio: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for src, trg, src_mask in train_loader:
        src = src.to(device)
        trg = trg.to(device)
        src_mask = src_mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs shape: (batch, trg_len, vocab_size)
        outputs, _ = model(src, trg, teacher_forcing_ratio, src_mask)

        # Reshape for cross-entropy: skip the first target token (BOS)
        # YOUR CODE HERE
        # 1. Reshape outputs[:, 1:, :] to (batch * (trg_len-1), vocab_size)
        # 2. Reshape trg[:, 1:] to (batch * (trg_len-1),)
        # 3. Compute cross-entropy loss, ignoring PAD_IDX
        # 4. Backpropagate and clip gradients
        # 5. Update parameters
        raise NotImplementedError("Implement the training step")

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


@torch.no_grad()
def validate(
    model: Seq2Seq,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss.

    Returns:
        Average validation loss.
    """
    model.eval()
    epoch_loss = 0.0
    num_batches = 0

    for src, trg, src_mask in val_loader:
        src = src.to(device)
        trg = trg.to(device)
        src_mask = src_mask.to(device)

        # Forward pass with no teacher forcing
        outputs, _ = model(src, trg, teacher_forcing_ratio=0.0, src_mask=src_mask)

        # Compute loss (same reshaping as training)
        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        trg_flat = trg[:, 1:].contiguous().view(-1)
        loss = criterion(outputs_flat, trg_flat)

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["data"]["random_seed"])
    logger = setup_logger(config["logging"]["log_dir"])
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Attention: {args.use_attention}")

    # Data
    data_cfg = config["data"]
    train_loader, val_loader, test_loader, src_c2i, trg_c2i, src_i2c, trg_i2c = (
        create_dataloaders(
            num_samples=data_cfg["num_samples"],
            train_size=data_cfg["train_size"],
            val_size=data_cfg["val_size"],
            test_size=data_cfg["test_size"],
            batch_size=config["training"]["batch_size"],
            seed=data_cfg["random_seed"],
        )
    )

    logger.info(f"Source vocab size: {len(src_c2i)}")
    logger.info(f"Target vocab size: {len(trg_c2i)}")

    # Model
    model = build_model(len(src_c2i), len(trg_c2i), config, args.use_attention, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    # Optimizer and loss
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Training loop
    tracker = MetricTracker()

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            train_cfg["teacher_forcing_ratio"],
            train_cfg["grad_clip_max_norm"],
            device,
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Track metrics
        tracker.update("train_loss", train_loss)
        tracker.update("val_loss", val_loss)

        logger.info(
            f"Epoch {epoch+1}/{train_cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(config["logging"]["save_dir"], "best_model.pt"),
                model, optimizer, epoch, best_val_loss,
            )
            logger.info(f"  -> New best model saved (val_loss={val_loss:.4f})")

    # Save final training curves
    tracker.plot(
        os.path.join(config["logging"]["log_dir"], "training_curves.png"),
        title="Seq2Seq Training Curves",
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
