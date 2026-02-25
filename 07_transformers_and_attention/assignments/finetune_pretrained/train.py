"""
Fine-Tune Pretrained Transformer - Training Script

Supports:
1. Fine-tuning BERT with discriminative LR and warmup
2. Training a small Transformer from scratch for comparison

Usage:
    python train.py --config config.yaml --task finetune_bert
    python train.py --config config.yaml --task from_scratch
    python train.py --config config.yaml --task data_efficiency
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from transformers import get_linear_schedule_with_warmup

from data import create_dataloaders
from model import BertClassifier, SmallTransformerClassifier
from utils import MetricTracker, setup_logger, save_checkpoint, set_seed, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained Transformer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--task",
        type=str,
        choices=["finetune_bert", "from_scratch", "data_efficiency"],
        required=True,
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_finetune_bert(config: dict, device: torch.device, logger):
    """Fine-tune BERT for sentiment classification."""
    ft_cfg = config["finetune"]
    model_cfg = config["model"]

    set_seed(ft_cfg["seed"])

    # Data
    train_loader, val_loader, tokenizer = create_dataloaders(
        dataset_name=config["data"]["dataset_name"],
        dataset_config=config["data"].get("dataset_config"),
        model_name=model_cfg["model_name"],
        max_length=model_cfg["max_length"],
        batch_size=ft_cfg["batch_size"],
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = BertClassifier(
        model_name=model_cfg["model_name"],
        num_labels=model_cfg["num_labels"],
    ).to(device)

    # Optimizer with discriminative learning rates
    param_groups = model.get_parameter_groups(ft_cfg["bert_lr"], ft_cfg["classifier_lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=ft_cfg["weight_decay"])

    # LR scheduler with warmup
    num_training_steps = ft_cfg["num_epochs"] * len(train_loader)
    num_warmup_steps = int(ft_cfg["warmup_fraction"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    criterion = nn.CrossEntropyLoss()
    tracker = MetricTracker()

    # Training loop
    for epoch in range(1, ft_cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs["logits"], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ft_cfg["grad_clip_max_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        tracker.update("train_loss", avg_loss)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        tracker.update("val_accuracy", val_metrics["accuracy"])
        tracker.update("val_f1", val_metrics["f1"])

        logger.info(
            f"Epoch {epoch}/{ft_cfg['num_epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

    save_checkpoint(
        os.path.join(config["logging"]["save_dir"], "bert_best.pt"),
        model, optimizer, ft_cfg["num_epochs"], val_metrics["accuracy"],
    )
    tracker.plot(os.path.join(config["logging"]["log_dir"], "finetune_curves.png"))
    logger.info("Fine-tuning complete!")


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset. Returns accuracy and F1."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs["logits"].argmax(dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return compute_metrics(all_preds, all_labels)


def train_from_scratch(config: dict, device: torch.device, logger):
    """Train a small Transformer from scratch for comparison."""
    # YOUR CODE HERE
    # 1. Build SmallTransformerClassifier
    # 2. Use a character-level or simple tokenizer
    # 3. Train with standard Adam optimizer
    # 4. Evaluate and compare with fine-tuned BERT
    raise NotImplementedError("Implement from-scratch training")


def run_data_efficiency_comparison(config: dict, device: torch.device, logger):
    """Compare fine-tuning vs from-scratch at different data sizes."""
    # YOUR CODE HERE
    # For each data size in [100, 500, 1000, 5000, full]:
    #   1. Fine-tune BERT
    #   2. Train from scratch
    #   3. Record accuracy for both
    # Plot accuracy vs data size for both approaches
    raise NotImplementedError("Implement data efficiency comparison")


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger = setup_logger(config["logging"]["log_dir"])

    logger.info(f"Device: {device}")
    logger.info(f"Task: {args.task}")

    if args.task == "finetune_bert":
        train_finetune_bert(config, device, logger)
    elif args.task == "from_scratch":
        train_from_scratch(config, device, logger)
    elif args.task == "data_efficiency":
        run_data_efficiency_comparison(config, device, logger)


if __name__ == "__main__":
    main()
