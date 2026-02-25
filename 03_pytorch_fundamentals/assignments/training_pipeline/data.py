"""Data loading pipeline for CIFAR-10.

Pre-written: uses shared_utils data loaders. Provides additional configuration
options beyond the defaults (custom augmentation, pin_memory, drop_last).
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# Add project root to path for shared_utils access
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.data import DATA_DIR


# CIFAR-10 channel-wise normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(augment: bool = True):
    """Get train and test transforms for CIFAR-10.

    Args:
        augment: If True, apply data augmentation to training transforms.

    Returns:
        Tuple of (train_transform, test_transform).
    """
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

    return train_transform, test_transform


def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
    augment: bool = True,
    pin_memory: bool = True,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders for CIFAR-10.

    The training set (50,000 images) is split into train (45,000) and
    validation (5,000) using a fixed seed for reproducibility.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers. Set to 0 on Windows
            if you encounter multiprocessing issues.
        val_split: Fraction of training data for validation.
        augment: Whether to apply data augmentation on training set.
        pin_memory: Pin memory for faster GPU transfer.
        data_dir: Directory to download/cache data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = data_dir or str(DATA_DIR)
    train_transform, test_transform = get_transforms(augment)

    # Download datasets
    full_train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform,
    )

    # Split training into train + validation
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# CIFAR-10 class names for display
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
