"""Data loading with configurable anti-patterns for profiling experiments."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

from shared_utils.data import DATA_DIR


def get_heavy_augmentation() -> transforms.Compose:
    """Return a deliberately heavy CPU augmentation pipeline (anti-pattern).

    Includes: RandomResizedCrop, ColorJitter, RandomRotation, RandomAffine,
    GaussianBlur, RandomErasing -- far more than necessary for CIFAR-10.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        transforms.RandomErasing(p=0.5),
    ])


def get_light_augmentation() -> transforms.Compose:
    """Return standard CIFAR-10 augmentation (efficient)."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])


def get_test_transform() -> transforms.Compose:
    """Return test/eval transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])


def load_cifar10_profiling(
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    heavy_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with configurable DataLoader settings for profiling.

    Args:
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster CPU-GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
        heavy_augmentation: If True, use heavy CPU augmentation (anti-pattern).

    Returns:
        Tuple of (train_loader, test_loader).
    """
    data_dir = str(DATA_DIR)

    train_transform = get_heavy_augmentation() if heavy_augmentation else get_light_augmentation()
    test_transform = get_test_transform()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    worker_kwargs = {}
    if persistent_workers and num_workers > 0:
        worker_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, **worker_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, test_loader
