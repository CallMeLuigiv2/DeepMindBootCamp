"""
Transformer from Scratch - Training Script

Trains either:
1. Encoder-decoder Transformer on number sorting
2. GPT decoder-only on character-level language modeling

Usage:
    python train.py --config config.yaml --task sorting
    python train.py --config config.yaml --task language_model
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml

from data import (
    create_sorting_dataloader,
    create_char_dataloaders,
    PAD_ID,
)
from model import Transformer, GPT
from utils import (
    MetricTracker,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_warmup_scheduler,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer from Scratch")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, choices=["sorting", "language_model"], required=True)
    parser.add_argument("--corpus", type=str, default=None, help="Text corpus path (for LM task)")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_sorting(config: dict, device: torch.device, logger):
    """Train encoder-decoder Transformer on number sorting."""
    sort_cfg = config["sorting"]
    t_cfg = config["transformer"]
    train_cfg = config["training"]

    # Data
    train_loader = create_sorting_dataloader(
        batch_size=train_cfg["batch_size"],
        seq_len_min=sort_cfg["seq_len_min"],
        seq_len_max=sort_cfg["seq_len_max"],
    )

    # Model
    model = Transformer(
        src_vocab_size=sort_cfg["vocab_size"],
        tgt_vocab_size=sort_cfg["vocab_size"],
        d_model=t_cfg["d_model"],
        num_heads=t_cfg["num_heads"],
        num_layers=t_cfg["num_layers"],
        d_ff=t_cfg["d_ff"],
        max_len=t_cfg["max_len"],
        dropout=t_cfg["dropout"],
    ).to(device)

    logger.info(f"Transformer parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = get_warmup_scheduler(optimizer, train_cfg["warmup_steps"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Training loop
    tracker = MetricTracker()
    model.train()
    data_iter = iter(train_loader)

    for step in range(1, sort_cfg["num_train_steps"] + 1):
        src, tgt = next(data_iter)
        src, tgt = src.to(device), tgt.to(device)

        # Create causal mask for decoder
        tgt_input = tgt[:, :-1]  # Shift right: remove last token
        tgt_target = tgt[:, 1:]  # Target: remove first token (SOS)
        tgt_mask = Transformer.generate_causal_mask(tgt_input.size(1), device)

        optimizer.zero_grad()

        # Forward pass
        # YOUR CODE HERE
        # 1. logits = model(src, tgt_input, tgt_mask=tgt_mask)
        # 2. Reshape logits and targets for cross-entropy
        # 3. Compute loss, backprop, clip gradients, step
        raise NotImplementedError("Implement sorting training step")

        scheduler.step()
        tracker.update("train_loss", loss.item())

        if step % config["logging"]["log_interval"] == 0:
            avg_loss = sum(tracker.get("train_loss")[-100:]) / min(100, step)
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Step {step}/{sort_cfg['num_train_steps']} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        # Evaluation
        if step % sort_cfg["eval_every"] == 0:
            accuracy = evaluate_sorting(model, device, sort_cfg)
            tracker.update("accuracy", accuracy)
            logger.info(f"  -> Sorting accuracy: {accuracy:.4f}")
            model.train()

    save_checkpoint(
        os.path.join(config["logging"]["save_dir"], "sorting_final.pt"),
        model, optimizer, sort_cfg["num_train_steps"], 0,
    )
    tracker.plot(os.path.join(config["logging"]["log_dir"], "sorting_curves.png"))


@torch.no_grad()
def evaluate_sorting(model, device, sort_cfg, num_samples=1000):
    """Evaluate sorting accuracy (exact match on full sequences)."""
    model.eval()
    correct = 0

    for _ in range(num_samples):
        import random
        length = random.randint(sort_cfg["seq_len_min"], sort_cfg["seq_len_max"])
        numbers = [random.randint(1, 100) for _ in range(length)]
        expected = sorted(numbers)

        src = torch.tensor([[101] + numbers + [102]], dtype=torch.long, device=device)
        # Autoregressive decoding
        # YOUR CODE HERE - greedy decode from the Transformer
        # Compare decoded output with expected sorted sequence
        pass

    return correct / num_samples


def train_language_model(config: dict, corpus_path: str, device: torch.device, logger):
    """Train GPT on character-level language modeling."""
    lm_cfg = config["language_model"]
    gpt_cfg = config["gpt"]
    train_cfg = config["training"]

    # Data
    train_loader, val_loader, dataset = create_char_dataloaders(
        corpus_path=corpus_path,
        context_length=lm_cfg["context_length"],
        batch_size=train_cfg["batch_size"],
    )
    logger.info(f"Vocabulary size: {dataset.vocab_size}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")

    # Model
    model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=gpt_cfg["d_model"],
        num_heads=gpt_cfg["num_heads"],
        num_layers=gpt_cfg["num_layers"],
        d_ff=gpt_cfg["d_ff"],
        max_len=gpt_cfg["max_len"],
        dropout=gpt_cfg["dropout"],
    ).to(device)

    logger.info(f"GPT parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = lm_cfg["num_epochs"] * len(train_loader)
    scheduler = get_warmup_scheduler(optimizer, train_cfg["warmup_steps"], total_steps)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    tracker = MetricTracker()
    global_step = 0

    for epoch in range(1, lm_cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # YOUR CODE HERE
            # 1. logits = model(x)
            # 2. Reshape for cross-entropy: (batch*seq_len, vocab) vs (batch*seq_len,)
            # 3. Compute loss, backprop, clip gradients, step
            raise NotImplementedError("Implement LM training step")

            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            # Generate samples periodically
            if global_step % lm_cfg["eval_every"] == 0:
                sample_text = generate_sample(model, dataset, device)
                logger.info(f"  Sample: {sample_text[:200]}")

        avg_loss = epoch_loss / len(train_loader)
        tracker.update("train_loss", avg_loss)

        # Validation
        val_loss = validate_lm(model, val_loader, criterion, device)
        tracker.update("val_loss", val_loss)

        logger.info(f"Epoch {epoch}/{lm_cfg['num_epochs']} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

    save_checkpoint(
        os.path.join(config["logging"]["save_dir"], "gpt_final.pt"),
        model, optimizer, lm_cfg["num_epochs"], 0,
    )
    tracker.plot(os.path.join(config["logging"]["log_dir"], "lm_curves.png"))


@torch.no_grad()
def validate_lm(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    return total_loss / len(val_loader)


@torch.no_grad()
def generate_sample(model, dataset, device, prompt_len=10, gen_len=100, temperature=0.8):
    model.eval()
    start = torch.randint(0, len(dataset.data) - prompt_len, (1,)).item()
    prompt = dataset.data[start : start + prompt_len].unsqueeze(0).to(device)
    generated = model.generate(prompt, max_new_tokens=gen_len, temperature=temperature)
    return dataset.decode(generated[0].tolist())


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger = setup_logger(config["logging"]["log_dir"])

    logger.info(f"Device: {device}")
    logger.info(f"Task: {args.task}")

    if args.task == "sorting":
        train_sorting(config, device, logger)
    else:
        corpus = args.corpus or config["language_model"]["corpus_path"]
        train_language_model(config, corpus, device, logger)


if __name__ == "__main__":
    main()
