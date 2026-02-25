"""Training script: baseline (naive) and optimized (all techniques combined).

Usage:
    python train.py --config config.yaml --mode baseline
    python train.py --config config.yaml --mode optimized
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from model import TransformerClassifier
from data import load_ag_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(config: dict, device: torch.device) -> None:
    """Train with naive (unoptimized) settings."""
    set_seed(42)
    bl_cfg = config["baseline"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    logger.info("Building model...")
    model = TransformerClassifier(
        vocab_size=model_cfg["vocab_size"],
        embedding_dim=model_cfg["embedding_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        ffn_hidden_dim=model_cfg["ffn_hidden_dim"],
        max_seq_length=model_cfg["max_seq_length"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        use_flash_attention=False,
        use_checkpointing=False,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    train_loader, val_loader, test_loader = load_ag_news(
        batch_size=bl_cfg["batch_size"],
        max_seq_length=model_cfg["max_seq_length"],
        vocab_size=model_cfg["vocab_size"],
        pre_tokenize=False,  # Slow: tokenize on-the-fly
        num_workers=bl_cfg["num_workers"],
        pin_memory=bl_cfg["pin_memory"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    training_logger = TrainingLogger()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in tqdm(train_loader, desc=f"Baseline Epoch {epoch}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        logger.info(f"Epoch {epoch}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")
        training_logger.log(train_loss=total_loss / total, train_acc=correct / total)

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, epoch, total_loss / total, str(save_dir / "baseline_final.pt"))


def train_optimized(config: dict, device: torch.device) -> None:
    """Train with all optimizations combined."""
    set_seed(42)
    opt_cfg = config["optimized"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # YOUR CODE HERE: Build the fully optimized pipeline
    # 1. Create model with flash_attention=True, checkpointing=True
    # 2. Compile model with torch.compile(mode=opt_cfg["compile_mode"])
    # 3. Load pre-tokenized data with optimal DataLoader settings
    # 4. Create GradScaler for mixed precision
    # 5. Training loop with:
    #    - autocast context manager
    #    - gradient accumulation
    #    - scaler.scale, scaler.step, scaler.update
    # 6. Measure throughput, memory, accuracy
    raise NotImplementedError("Implement fully optimized training pipeline")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Optimization")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "optimized"])
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "baseline":
        train_baseline(config, device)
    elif args.mode == "optimized":
        train_optimized(config, device)


if __name__ == "__main__":
    main()
