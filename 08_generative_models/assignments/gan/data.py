"""
GAN - Data Loading

Loads MNIST, CIFAR-10, or CelebA with GAN-appropriate transforms
(normalize to [-1, 1] for Tanh output).
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> DataLoader:
    """Load MNIST normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def get_cifar10_dataloader(
    batch_size: int = 128,
    image_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> DataLoader:
    """Load CIFAR-10 resized and normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def get_celeba_dataloader(
    batch_size: int = 128,
    image_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> DataLoader:
    """Load CelebA cropped and resized, normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CelebA(root=data_dir, split="train", download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def get_dataloader(
    dataset_name: str = "mnist",
    batch_size: int = 128,
    image_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> DataLoader:
    """Get the appropriate dataloader by name."""
    loaders = {
        "mnist": lambda: get_mnist_dataloader(batch_size, data_dir, num_workers),
        "cifar10": lambda: get_cifar10_dataloader(batch_size, image_size, data_dir, num_workers),
        "celeba": lambda: get_celeba_dataloader(batch_size, image_size, data_dir, num_workers),
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    return loaders[dataset_name]()
