"""Data loading for the Classic Architectures project.

Pre-written: CIFAR-10 and MNIST loading with standard augmentation.
Uses shared_utils where appropriate.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.data import DATA_DIR


# Normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
    augment: bool = True,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 with standard augmentation for fair model comparison.

    Augmentation (matching the original VGG/ResNet training):
        Training: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
        Test/Val: Normalize only

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Data loader workers.
        val_split: Fraction of training data for validation.
        augment: Whether to augment training data.
        data_dir: Data cache directory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = data_dir or str(DATA_DIR)
    normalize = T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), normalize])

    test_transform = T.Compose([T.ToTensor(), normalize])

    full_train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform,
    )

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_mnist_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST with padding to 32x32 for LeNet-5 (Part 1).

    Args:
        batch_size: Batch size.
        num_workers: Data loader workers.
        data_dir: Data cache directory.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    root = data_dir or str(DATA_DIR)
    transform = T.Compose([
        T.Pad(2),  # 28x28 -> 32x32
        T.ToTensor(),
        T.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


# Class names for display
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
