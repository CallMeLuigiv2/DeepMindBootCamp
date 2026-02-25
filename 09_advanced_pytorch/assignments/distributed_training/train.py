"""Training script for distributed training and advanced techniques.

Supports three modes:
- single: Standard single-GPU training (baseline)
- ddp: DistributedDataParallel training
- full: Combined pipeline (DDP + mixed precision + gradient accumulation +
        gradient checkpointing + torch.compile)

Usage:
    python train.py --config config.yaml --mode single
    torchrun --nproc_per_node=2 train.py --config config.yaml --mode ddp
    torchrun --nproc_per_node=2 train.py --config config.yaml --mode full
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, save_checkpoint, TrainingLogger

from model import ResNet18CIFAR
from data import load_cifar10_distributed
from utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    print_rank0,
    ThroughputTimer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_single_gpu(config: dict) -> None:
    """Standard single-GPU training baseline."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Single-GPU training on {device}")

    model = ResNet18CIFAR(num_classes=config["model"]["num_classes"]).to(device)
    train_loader, val_loader, test_loader, _ = load_cifar10_distributed(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augment=config["data"]["augment"],
        distributed=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    training_logger = TrainingLogger()
    timer = ThroughputTimer()

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            timer.start()
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            timer.stop(data.size(0))
            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        throughput = timer.throughput()
        timer.reset()

        training_logger.log(train_loss=train_loss, train_acc=train_acc, throughput=throughput)
        logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}, throughput={throughput:.1f} samp/s")

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, epoch, train_loss, str(save_dir / "single_gpu_final.pt"))


def train_ddp(config: dict) -> None:
    """DistributedDataParallel training.

    Launch with: torchrun --nproc_per_node=N train.py --config config.yaml --mode ddp
    """
    rank, local_rank, world_size = setup_distributed(backend=config["distributed"]["backend"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(42 + rank)

    print_rank0(f"DDP training: world_size={world_size}")

    model = ResNet18CIFAR(num_classes=config["model"]["num_classes"]).to(device)

    # YOUR CODE HERE
    # 1. Wrap model in DDP: model = DDP(model, device_ids=[local_rank])
    raise NotImplementedError("Wrap model with DDP")

    train_loader, val_loader, test_loader, train_sampler = load_cifar10_distributed(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augment=config["data"]["augment"],
        distributed=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        # YOUR CODE HERE
        # 1. Set epoch on train_sampler: train_sampler.set_epoch(epoch)
        raise NotImplementedError("Set sampler epoch for DDP")

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        if is_main_process():
            logger.info(f"Epoch {epoch}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

    # Save checkpoint from rank 0 only
    if is_main_process():
        save_dir = Path(config["logging"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model.module, optimizer, epoch, total_loss / total,
            str(save_dir / "ddp_final.pt"),
        )

    cleanup_distributed()


def train_full_pipeline(config: dict) -> None:
    """Full combined pipeline: DDP + mixed precision + gradient accumulation +
    gradient checkpointing + torch.compile.

    Launch with: torchrun --nproc_per_node=N train.py --config config.yaml --mode full
    """
    rank, local_rank, world_size = setup_distributed(backend=config["distributed"]["backend"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(42 + rank)

    print_rank0("Full pipeline: DDP + AMP + Accumulation + Checkpointing + Compile")

    accum_cfg = config["accumulation"]
    mp_cfg = config["mixed_precision"]
    compile_cfg = config["compile"]
    ckpt_cfg = config["checkpointing"]

    # YOUR CODE HERE: Build the full pipeline
    # 1. Create model with gradient checkpointing if enabled
    # 2. Wrap with DDP
    # 3. Compile with torch.compile if enabled
    # 4. Create GradScaler for mixed precision if enabled
    # 5. Training loop with:
    #    - autocast context manager
    #    - gradient accumulation with model.no_sync() for intermediate steps
    #    - scaler.scale(loss / accum_steps).backward()
    #    - scaler.step(optimizer) and scaler.update() on accumulation boundary
    # 6. Logging from rank 0 only
    # 7. Checkpoint saving from rank 0 only
    raise NotImplementedError("Implement full combined pipeline")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "ddp", "full"])
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.mode == "single":
        train_single_gpu(config)
    elif args.mode == "ddp":
        train_ddp(config)
    elif args.mode == "full":
        train_full_pipeline(config)


if __name__ == "__main__":
    main()
