"""
VAE - Data Loading

Loads MNIST dataset with appropriate transforms for the VAE.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create MNIST train and test DataLoaders.

    Images are normalized to [0, 1] (suitable for BCE reconstruction loss).

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # ToTensor() already scales to [0, 1] — no additional normalization needed
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, test_loader


def flatten_batch(images: torch.Tensor) -> torch.Tensor:
    """Flatten a batch of MNIST images from (B, 1, 28, 28) to (B, 784)."""
    return images.view(images.size(0), -1)


def unflatten_batch(flat_images: torch.Tensor) -> torch.Tensor:
    """Reshape a batch from (B, 784) to (B, 1, 28, 28)."""
    return flat_images.view(flat_images.size(0), 1, 28, 28)
