"""Complete training pipeline for CIFAR-10.

Pre-written: argparse, config loading, device setup, optimizer/scheduler creation,
             checkpoint saving/loading, TensorBoard logging, early stopping, summary.
Stubbed: train_one_epoch and validate loop bodies.

Usage:
    python train.py --config config.yaml
    python train.py --epochs 100 --batch-size 128 --lr 0.01 --optimizer adamw
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

# Add project root to path for shared_utils access
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import set_seed, get_device, save_checkpoint, load_checkpoint, EarlyStopping

from model import CIFAR10Net
from data import get_data_loaders


# ============================================================
# Argument Parsing (Pre-written)
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw"], default=None)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step", "onecycle", "plateau"], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--num-workers", type=int, default=None, help="Data loader workers")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm (0=disable)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Load YAML config and override with CLI arguments."""
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"training": {}, "data": {}, "model": {}, "output": {}, "scheduler_config": {}}

    # CLI overrides
    t = config.setdefault("training", {})
    d = config.setdefault("data", {})
    o = config.setdefault("output", {})
    if args.epochs is not None: t["epochs"] = args.epochs
    if args.batch_size is not None: t["batch_size"] = args.batch_size
    if args.lr is not None: t["learning_rate"] = args.lr
    if args.weight_decay is not None: t["weight_decay"] = args.weight_decay
    if args.optimizer is not None: t["optimizer"] = args.optimizer
    if args.scheduler is not None: t["scheduler"] = args.scheduler
    if args.patience is not None: t["patience"] = args.patience
    if args.grad_clip is not None: t["grad_clip"] = args.grad_clip
    if args.num_workers is not None: d["num_workers"] = args.num_workers
    if args.seed is not None: o["seed"] = args.seed
    if args.save_dir is not None: o["save_dir"] = args.save_dir
    if args.log_dir is not None: o["log_dir"] = args.log_dir

    return config


# ============================================================
# Optimizer & Scheduler Creation (Pre-written)
# ============================================================

def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    t = config["training"]
    lr = t.get("learning_rate", 0.01)
    wd = t.get("weight_decay", 1e-4)
    opt_name = t.get("optimizer", "adamw").lower()

    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer, train_loader=None):
    """Create learning rate scheduler from config."""
    t = config["training"]
    sc = config.get("scheduler_config", {})
    sched_name = t.get("scheduler", "cosine").lower()

    if sched_name == "cosine":
        sc_cfg = sc.get("cosine", {})
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sc_cfg.get("T_max", t.get("epochs", 100)),
            eta_min=sc_cfg.get("eta_min", 1e-4),
        )
    elif sched_name == "step":
        sc_cfg = sc.get("step", {})
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sc_cfg.get("step_size", 30),
            gamma=sc_cfg.get("gamma", 0.1),
        )
    elif sched_name == "plateau":
        sc_cfg = sc.get("plateau", {})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=sc_cfg.get("factor", 0.5),
            patience=sc_cfg.get("patience", 5),
            min_lr=sc_cfg.get("min_lr", 1e-5),
        )
    elif sched_name == "onecycle":
        sc_cfg = sc.get("onecycle", {})
        if train_loader is None:
            raise ValueError("OneCycleLR requires train_loader")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sc_cfg.get("max_lr", t.get("learning_rate", 0.01)),
            epochs=t.get("epochs", 100),
            steps_per_epoch=len(train_loader),
            pct_start=sc_cfg.get("pct_start", 0.3),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ============================================================
# Training and Validation Loops (Stubbed)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    scheduler=None,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The CNN model.
        loader: Training DataLoader.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer.
        device: Device to run on.
        grad_clip: Max gradient norm. 0 disables clipping.
        scheduler: If OneCycleLR, step per batch.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # YOUR CODE HERE:
        # 1. Zero gradients
        # 2. Forward pass
        # 3. Compute loss
        # 4. Backward pass
        # 5. Gradient clipping (if grad_clip > 0)
        # 6. Optimizer step
        # 7. Update running_loss, correct, total
        # 8. If scheduler is OneCycleLR, call scheduler.step()
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
        model: The CNN model.
        loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Device to run on.

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
# Main Training Function (Pre-written scaffolding)
# ============================================================

def main():
    args = parse_args()
    config = load_config(args)

    # Extract config values
    t = config["training"]
    d = config["data"]
    o = config["output"]

    seed = o.get("seed", 42)
    epochs = t.get("epochs", 100)
    batch_size = t.get("batch_size", 128)
    patience = t.get("patience", 15)
    grad_clip = t.get("grad_clip", 1.0)
    save_dir = o.get("save_dir", "checkpoints")
    log_dir = o.get("log_dir", "runs")
    sched_name = t.get("scheduler", "cosine").lower()

    # Setup
    set_seed(seed)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("CIFAR-10 TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Seed:       {seed}")
    print(f"Epochs:     {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer:  {t.get('optimizer', 'adamw')}")
    print(f"Scheduler:  {sched_name}")
    print(f"LR:         {t.get('learning_rate', 0.01)}")
    print()

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=d.get("num_workers", 2),
        val_split=d.get("val_split", 0.1),
        augment=d.get("augment", True),
        pin_memory=d.get("pin_memory", True) and device.type == "cuda",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Model
    model = CIFAR10Net(num_classes=config.get("model", {}).get("num_classes", 10)).to(device)
    param_count = model.count_parameters()
    print(f"\nModel parameters: {param_count:,}")
    if param_count > 500_000:
        print("WARNING: Model exceeds 500K parameter budget!")
    print()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer, train_loader if sched_name == "onecycle" else None)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    best_epoch = start_epoch
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        is_onecycle = sched_name == "onecycle"
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=grad_clip,
            scheduler=scheduler if is_onecycle else None,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step (non-OneCycleLR)
        if not is_onecycle:
            if sched_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                path=os.path.join(save_dir, "best_model.pth"),
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
            )

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            path=os.path.join(save_dir, "latest_model.pth"),
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
        )

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time
    writer.close()

    # ------------------------------------------------------------------
    # Final Evaluation on Test Set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)
        print(f"Loaded best model from epoch {best_epoch + 1}")

    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best epoch:          {best_epoch + 1}")
    print(f"Best val accuracy:   {best_val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Parameters:          {param_count:,}")

    if test_acc >= 0.85:
        print("\nTarget reached: 85%+ test accuracy!")
    else:
        print(f"\nBelow target: {test_acc:.1%} < 85%. See debugging tips in README.")


if __name__ == "__main__":
    main()
