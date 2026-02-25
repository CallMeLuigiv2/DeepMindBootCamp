"""Training with configurable precision: FP32, FP16, BF16.

Usage:
    python train.py --config config.yaml --precision fp32
    python train.py --config config.yaml --precision fp16
    python train.py --config config.yaml --precision bf16
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from shared_utils.common import set_seed, get_device, save_checkpoint, TrainingLogger

from model import ResNet18ForQuantization
from data import load_cifar10_standard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler=None,
    autocast_dtype=None,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        scaler: GradScaler for FP16 (None for FP32/BF16).
        autocast_dtype: dtype for autocast (None for FP32).

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()

        if autocast_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(data)
                loss = criterion(output, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    autocast_dtype=None,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if autocast_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(data)
                loss = criterion(output, target)
        else:
            output = model(data)
            loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Mixed Precision Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    set_seed(42)
    logger.info(f"Training with {args.precision.upper()} on {device}")

    model = ResNet18ForQuantization(num_classes=config["model"]["num_classes"]).to(device)
    train_loader, val_loader, test_loader = load_cifar10_standard(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"],
    )
    criterion = nn.CrossEntropyLoss()

    # Configure precision
    autocast_dtype = None
    scaler = None
    if args.precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler()
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        # No GradScaler needed for BF16

    # Track GradScaler scale factor
    training_logger = TrainingLogger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    import time
    start_time = time.perf_counter()

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, autocast_dtype,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, autocast_dtype)
        scheduler.step()

        log_data = dict(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        if scaler is not None:
            log_data["grad_scale"] = scaler.get_scale()
        training_logger.log(**log_data)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    total_time = time.perf_counter() - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, autocast_dtype)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Total training time: {total_time:.1f}s")
    logger.info(f"Peak GPU memory: {peak_mem:.1f} MB")

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model, optimizer, config["training"]["num_epochs"], train_loss,
        str(save_dir / f"{args.precision}_final.pt"),
        test_acc=test_acc,
    )
    training_logger.save(str(save_dir / f"{args.precision}_history.json"))


if __name__ == "__main__":
    main()
