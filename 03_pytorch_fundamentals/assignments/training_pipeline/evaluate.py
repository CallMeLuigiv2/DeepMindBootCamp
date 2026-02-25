"""Standalone evaluation script for CIFAR-10 trained models.

Pre-written: checkpoint loading, data loading, overall accuracy.
Stubbed: per-class accuracy computation, confusion matrix visualization.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import get_device, load_checkpoint
from shared_utils.plotting import plot_confusion_matrix

from model import CIFAR10Net
from data import get_data_loaders, CIFAR10_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Data loader workers",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Directory to save plots",
    )
    return parser.parse_args()


def compute_per_class_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class accuracy and collect all predictions.

    Args:
        model: Trained model.
        loader: Test DataLoader.
        device: Device to run on.
        num_classes: Number of classes.

    Returns:
        Tuple of (all_preds, all_labels, per_class_accuracy).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # YOUR CODE HERE: compute per-class accuracy
    # Hint: for each class, compute (correct predictions for that class) / (total samples of that class)
    per_class_acc = np.zeros(num_classes)
    # YOUR CODE HERE
    pass

    return all_preds, all_labels, per_class_acc


def print_confusion_matrix_table(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    class_names: list[str],
) -> None:
    """Print confusion matrix as a formatted text table.

    Args:
        all_preds: Predicted labels.
        all_labels: True labels.
        class_names: List of class names.
    """
    # YOUR CODE HERE
    # Hint: use sklearn.metrics.confusion_matrix or compute manually
    pass


def main():
    args = parse_args()

    # Setup
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CIFAR10Net().to(device)
    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best val loss: {ckpt.get('best_val_loss', '?'):.4f}")
    print(f"  Best val acc:  {ckpt.get('best_val_acc', '?'):.4f}")
    print(f"  Parameters:    {model.count_parameters():,}")

    # Load test data
    _, _, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    # Overall accuracy
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    test_loss /= len(test_loader)

    print(f"\nOverall Test Accuracy: {test_acc:.4f} ({correct}/{total})")
    print(f"Overall Test Loss:    {test_loss:.4f}")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    all_preds, all_labels, per_class_acc = compute_per_class_accuracy(
        model, test_loader, device,
    )

    for i, name in enumerate(CIFAR10_CLASSES):
        acc = per_class_acc[i] if per_class_acc is not None else 0.0
        print(f"  {name:<15} {acc:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print_confusion_matrix_table(all_preds, all_labels, CIFAR10_CLASSES)

    # Save confusion matrix plot
    if all_preds is not None and all_labels is not None:
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(all_labels, all_preds)
        plot_confusion_matrix(
            cm, class_names=CIFAR10_CLASSES,
            title="CIFAR-10 Test Confusion Matrix",
            save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
        )
        print(f"\nConfusion matrix saved to {args.output_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
