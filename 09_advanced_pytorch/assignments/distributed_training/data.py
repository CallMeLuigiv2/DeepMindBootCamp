"""Data loading for distributed training with DistributedSampler support."""

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from shared_utils.data import load_cifar10


def load_cifar10_distributed(
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    augment: bool = True,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler]]:
    """Load CIFAR-10 with optional DistributedSampler for DDP.

    Args:
        batch_size: Per-GPU batch size.
        val_split: Fraction of training data for validation.
        num_workers: Number of data loading workers.
        augment: Whether to apply data augmentation.
        distributed: Whether to use DistributedSampler.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_sampler).
        train_sampler is None if not distributed.
    """
    if not distributed:
        train_loader, val_loader, test_loader = load_cifar10(
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            augment=augment,
        )
        return train_loader, val_loader, test_loader, None

    # For distributed: need to create datasets manually and wrap with DistributedSampler
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import random_split
    from pathlib import Path

    data_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data")
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_sampler
