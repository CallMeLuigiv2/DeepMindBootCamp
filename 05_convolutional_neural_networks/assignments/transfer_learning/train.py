"""Training script for the Transfer Learning project.

Pre-written: argparse, config loading, strategy selection, two-phase training
             scaffolding, logging, checkpointing.
Stubbed: training loop bodies, two-phase transition logic.

Usage:
    python train.py --config config.yaml --strategy frozen
    python train.py --config config.yaml --strategy full
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

from shared_utils.common import set_seed, get_device, save_checkpoint, load_checkpoint, EarlyStopping

from model import (
    create_model,
    count_trainable,
    freeze_backbone,
    unfreeze_all,
    get_partial_param_groups,
    get_differential_param_groups,
)
from data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer Learning Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--strategy", type=str, default="frozen",
        choices=["scratch", "frozen", "partial", "full"],
        help="Transfer learning strategy",
    )
    parser.add_argument("--compare-all", action="store_true", help="Train all strategies")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", type=str, default=None,
                        choices=["minimal", "standard", "aggressive"])
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        model: The model.
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
        model: The model.
        loader: Validation DataLoader.
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
# Strategy-Specific Training
# ============================================================

def train_strategy(
    strategy: str,
    config: dict,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
) -> dict:
    """Train a model with the given strategy.

    Handles the two-phase training for Strategy 3 (full fine-tuning).

    Args:
        strategy: Strategy name.
        config: Full config dict.
        device: Device.
        train_loader, val_loader, test_loader: Data loaders.
        num_classes: Number of classes.

    Returns:
        Results dictionary.
    """
    strat_config = config["strategies"].get(strategy, {})
    model_config = config.get("model", {})
    save_dir = config["output"].get("save_dir", "checkpoints")
    log_dir = config["output"].get("log_dir", "runs")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'=' * 60}")
    print(f"  {strat_config.get('description', strategy)}")

    # Create model
    model = create_model(strategy, num_classes, {
        "backbone": model_config.get("backbone", "resnet50"),
        "unfreeze_from": strat_config.get("unfreeze_from", "layer4"),
    }).to(device)

    trainable, total = count_trainable(model)
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(log_dir, strategy))
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    if strategy == "full":
        # Two-phase training for Strategy 3
        warmup_epochs = strat_config.get("warmup_epochs", 5)
        finetune_epochs = strat_config.get("finetune_epochs", 25)

        # Phase 1: Warmup -- train head only
        print(f"\n  Phase 1: Head warmup ({warmup_epochs} epochs)")
        freeze_backbone(model)
        warmup_lr = strat_config.get("warmup_lr", 1e-3)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=warmup_lr,
        )

        for epoch in range(warmup_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            print(f"    Epoch {epoch + 1}/{warmup_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f}")

        # Phase 2: Full fine-tuning with differential LR
        print(f"\n  Phase 2: Full fine-tuning ({finetune_epochs} epochs)")
        unfreeze_all(model)
        param_groups = get_differential_param_groups(model, strat_config)

        if param_groups:
            optimizer = torch.optim.Adam(param_groups)
        else:
            # Fallback if not implemented yet
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs,
        )

        for epoch in range(finetune_epochs):
            global_epoch = warmup_epochs + epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            writer.add_scalar("Loss/train", train_loss, global_epoch)
            writer.add_scalar("Loss/val", val_loss, global_epoch)
            writer.add_scalar("Accuracy/train", train_acc, global_epoch)
            writer.add_scalar("Accuracy/val", val_acc, global_epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = global_epoch
                save_checkpoint(
                    model, optimizer, global_epoch, val_loss,
                    path=os.path.join(save_dir, f"{strategy}_best.pth"),
                    best_val_acc=best_val_acc,
                    strategy=strategy,
                )

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {global_epoch + 1} | "
                      f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                      f"Val: {val_loss:.4f}/{val_acc:.4f}")

    else:
        # Standard training for scratch, frozen, partial
        epochs = strat_config.get("epochs", 30)
        lr = strat_config.get("learning_rate", 1e-3)

        # Build optimizer with appropriate param groups
        if strategy == "partial":
            param_groups = get_partial_param_groups(model, strat_config)
            if param_groups:
                optimizer = torch.optim.Adam(param_groups)
            else:
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=lr,
                )
        else:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=lr,
            )

        sched_name = strat_config.get("scheduler", "step")
        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=strat_config.get("step_size", 10),
                gamma=strat_config.get("gamma", 0.5),
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs,
            )

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    path=os.path.join(save_dir, f"{strategy}_best.pth"),
                    best_val_acc=best_val_acc,
                    strategy=strategy,
                )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{epochs} | "
                      f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                      f"Val: {val_loss:.4f}/{val_acc:.4f}")

    total_time = time.time() - start_time
    writer.close()

    # Test evaluation
    best_path = os.path.join(save_dir, f"{strategy}_best.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    test_loss, test_acc = validate(model, test_loader, criterion, device)

    results = {
        "strategy": strategy,
        "trainable_params": trainable,
        "total_params": total,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "best_epoch": best_epoch + 1,
        "total_time": total_time,
    }

    print(f"\n  Best val acc:  {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Time: {total_time:.1f}s")

    return results


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.augmentation:
        config.setdefault("data", {})["augmentation_pipeline"] = args.augmentation

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    data_config = config.get("data", {})
    data_config.setdefault("batch_size", 32)
    train_loader, val_loader, test_loader, class_names, num_classes = load_dataset(data_config)
    print(f"Dataset: {data_config.get('dataset', 'unknown')}")
    print(f"Classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    config.setdefault("model", {})["num_classes"] = num_classes

    if args.compare_all:
        all_results = {}
        for strategy in ["scratch", "frozen", "partial", "full"]:
            set_seed(args.seed)
            results = train_strategy(
                strategy, config, device,
                train_loader, val_loader, test_loader, num_classes,
            )
            all_results[strategy] = results

        # Print comparison
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        print(f"{'Strategy':<12} {'Trainable':>12} {'Total':>12} {'Val Acc':>10} {'Test Acc':>10} {'Time':>10}")
        print("-" * 80)
        for s, r in all_results.items():
            print(
                f"{s:<12} {r['trainable_params']:>12,} {r['total_params']:>12,} "
                f"{r['best_val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['total_time']:>9.1f}s"
            )
    else:
        train_strategy(
            args.strategy, config, device,
            train_loader, val_loader, test_loader, num_classes,
        )


if __name__ == "__main__":
    main()
