"""
Diffusion Model - Data Loading

Loads MNIST or Fashion-MNIST with transforms suitable for diffusion models.
Images are normalized to [-1, 1] (matching Gaussian noise distribution).
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(
    dataset_name: str = "mnist",
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 0,
    train: bool = True,
) -> DataLoader:
    """Load image dataset normalized to [-1, 1].

    Args:
        dataset_name: "mnist" or "fashion_mnist"
        batch_size: Batch size
        data_dir: Directory for downloaded data
        num_workers: DataLoader workers
        train: Whether to load training or test split

    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Map [0, 1] -> [-1, 1]
    ])

    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform,
        )
    elif dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir, train=train, download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, drop_last=train,
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] back to [0, 1] for display."""
    return (images + 1) / 2
