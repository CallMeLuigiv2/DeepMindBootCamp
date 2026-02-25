"""Training script for the Classic Architectures project.

Pre-written: argparse, config loading, optimizer/scheduler setup, logging,
             checkpointing, comparison mode.
Stubbed: train_one_epoch and validate loop bodies.

Usage:
    python train.py --config config.yaml --arch resnet18
    python train.py --config config.yaml --compare-all
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import set_seed, get_device, save_checkpoint, load_checkpoint

from model import create_model
from data import get_cifar10_loaders, get_mnist_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Classic CNN Architectures")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--arch", type=str, default="resnet18",
        choices=["lenet5", "vgg11", "vgg11_fc", "resnet18", "plainnet18"],
        help="Architecture to train",
    )
    parser.add_argument("--compare-all", action="store_true", help="Train and compare all architectures")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    t = config["training"]
    lr = t.get("learning_rate", 0.1)
    wd = t.get("weight_decay", 5e-4)
    momentum = t.get("momentum", 0.9)
    opt_name = t.get("optimizer", "sgd").lower()

    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    t = config["training"]
    sched_name = t.get("scheduler", "multistep").lower()

    if sched_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=t.get("milestones", [80, 120]),
            gamma=t.get("gamma", 0.1),
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.get("epochs", 150),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ============================================================
# Training and Validation (Stubbed)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: CNN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # YOUR CODE HERE:
        # 1. Zero gradients
        # 2. Forward pass
        # 3. Compute loss
        # 4. Backward pass
        # 5. Optimizer step
        # 6. Update running_loss, correct, total
        pass

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: CNN model.
        loader: Validation/test DataLoader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # YOUR CODE HERE:
            # 1. Forward pass
            # 2. Compute loss
            # 3. Update running_loss, correct, total
            pass

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ============================================================
# Main
# ============================================================

def train_architecture(arch: str, config: dict, device: torch.device):
    """Train a single architecture and return results."""
    t = config["training"]
    o = config["output"]
    arch_config = config.get("architectures", {}).get(arch, {})

    epochs = t.get("epochs", 150)
    batch_size = t.get("batch_size", 128)
    save_dir = o.get("save_dir", "checkpoints")
    log_dir = o.get("log_dir", "runs")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {arch.upper()}")
    print(f"{'=' * 60}")
    print(f"  Config: {arch_config.get('description', arch)}")

    # Data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=config["data"].get("num_workers", 2),
        val_split=config["data"].get("val_split", 0.1),
        augment=config["data"].get("augment", True),
    )

    # Model
    model = create_model(arch, arch_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, arch))

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"Train: {train_loss:.4f} / {train_acc:.4f} | "
                f"Val: {val_loss:.4f} / {val_acc:.4f} | "
                f"LR: {current_lr:.5f}"
            )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                path=os.path.join(save_dir, f"{arch}_best.pth"),
                best_val_acc=best_val_acc,
                arch=arch,
            )

    total_time = time.time() - start_time
    writer.close()

    # Final test evaluation
    best_path = os.path.join(save_dir, f"{arch}_best.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    test_loss, test_acc = validate(model, test_loader, criterion, device)

    results = {
        "arch": arch,
        "params": param_count,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "total_time": total_time,
        "time_per_epoch": total_time / epochs,
    }

    print(f"\n  Best val acc:  {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Total time:    {total_time:.1f}s ({total_time / 60:.1f} min)")

    return results


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.lr: config["training"]["learning_rate"] = args.lr
    if args.num_workers: config["data"]["num_workers"] = args.num_workers

    seed = args.seed or config["training"].get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    if args.compare_all:
        # Train all architectures for comparison (Part 4)
        architectures = ["lenet5", "vgg11", "resnet18", "plainnet18"]
        all_results = {}

        for arch in architectures:
            set_seed(seed)  # Reset seed for fair comparison
            results = train_architecture(arch, config, device)
            all_results[arch] = results

        # Print comparison table
        print("\n" + "=" * 90)
        print("HEAD-TO-HEAD COMPARISON")
        print("=" * 90)
        print(f"{'Model':<15} {'Params':>12} {'Test Acc':>10} {'Best Val':>10} {'Time':>10} {'Time/Ep':>10}")
        print("-" * 90)
        for arch, r in all_results.items():
            print(
                f"{arch:<15} {r['params']:>12,} {r['test_acc']:>10.4f} "
                f"{r['best_val_acc']:>10.4f} {r['total_time']:>9.1f}s {r['time_per_epoch']:>9.1f}s"
            )
    else:
        # Train a single architecture
        train_architecture(args.arch, config, device)


if __name__ == "__main__":
    main()
