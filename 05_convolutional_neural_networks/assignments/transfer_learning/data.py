"""Data loading for the Transfer Learning project.

Pre-written: dataset downloading, ImageNet normalization, augmentation pipelines,
             train/val/test splitting.
Stubbed: custom dataset class if using non-torchvision datasets.
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as T


# ImageNet normalization (ALWAYS use when using ImageNet-pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Augmentation Pipelines (Pre-written)
# ============================================================

def get_transforms(pipeline: str = "standard", image_size: int = 224):
    """Get train and val/test transforms for transfer learning.

    Args:
        pipeline: Augmentation pipeline name ('minimal', 'standard', 'aggressive').
        image_size: Target image size (default: 224 for ImageNet models).

    Returns:
        Tuple of (train_transform, val_transform).
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])

    if pipeline == "minimal":
        train_transform = val_transform  # No augmentation

    elif pipeline == "standard":
        train_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    elif pipeline == "aggressive":
        train_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.25),
        ])
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}. Use 'minimal', 'standard', or 'aggressive'.")

    return train_transform, val_transform


# ============================================================
# Dataset Loaders (Pre-written)
# ============================================================

def load_flowers102(
    data_dir: str = "data",
    image_size: int = 224,
    pipeline: str = "standard",
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], int]:
    """Load Oxford Flowers 102 dataset.

    This dataset has pre-defined train/val/test splits.

    Args:
        data_dir: Directory to download/cache data.
        image_size: Target image size.
        pipeline: Augmentation pipeline name.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, num_classes).
    """
    train_transform, val_transform = get_transforms(pipeline, image_size)

    train_dataset = torchvision.datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=train_transform,
    )
    val_dataset = torchvision.datasets.Flowers102(
        root=data_dir, split="val", download=True, transform=val_transform,
    )
    test_dataset = torchvision.datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    num_classes = 102
    class_names = [f"flower_{i}" for i in range(num_classes)]

    return train_loader, val_loader, test_loader, class_names, num_classes


def load_eurosat(
    data_dir: str = "data",
    image_size: int = 224,
    pipeline: str = "standard",
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], int]:
    """Load EuroSAT dataset (satellite imagery, 10 classes).

    Args:
        data_dir: Directory to download/cache data.
        image_size: Target image size.
        pipeline: Augmentation pipeline name.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        seed: Random seed for splitting.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, num_classes).
    """
    train_transform, val_transform = get_transforms(pipeline, image_size)

    full_dataset = torchvision.datasets.EuroSAT(
        root=data_dir, download=True, transform=train_transform,
    )

    n = len(full_dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    ]
    num_classes = 10

    return train_loader, val_loader, test_loader, class_names, num_classes


def load_imagefolder(
    root_dir: str,
    image_size: int = 224,
    pipeline: str = "standard",
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], int]:
    """Load a custom dataset organized as an ImageFolder.

    Expected directory structure:
        root_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                ...

    Args:
        root_dir: Path to the ImageFolder root.
        image_size: Target image size.
        pipeline: Augmentation pipeline name.
        batch_size: Batch size.
        num_workers: Data loader workers.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        seed: Random seed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, num_classes).
    """
    train_transform, val_transform = get_transforms(pipeline, image_size)

    full_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    n = len(full_dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names, num_classes


# ============================================================
# Dataset Factory
# ============================================================

def load_dataset(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, list[str], int]:
    """Load dataset based on config.

    Args:
        config: Data configuration dict from config.yaml.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, num_classes).
    """
    name = config.get("dataset", "flowers102")
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 2)
    pipeline = config.get("augmentation_pipeline", "standard")

    if name == "flowers102":
        return load_flowers102(
            image_size=image_size, pipeline=pipeline,
            batch_size=batch_size, num_workers=num_workers,
        )
    elif name == "eurosat":
        return load_eurosat(
            image_size=image_size, pipeline=pipeline,
            batch_size=batch_size, num_workers=num_workers,
            seed=config.get("seed", 42),
        )
    elif os.path.isdir(name):
        return load_imagefolder(
            root_dir=name, image_size=image_size, pipeline=pipeline,
            batch_size=batch_size, num_workers=num_workers,
            seed=config.get("seed", 42),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            "Use 'flowers102', 'eurosat', or a path to an ImageFolder directory."
        )


# ============================================================
# Image Unnormalization (for visualization)
# ============================================================

def unnormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for display.

    Args:
        tensor: Normalized image tensor of shape (C, H, W).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Unnormalized numpy array of shape (H, W, C) in [0, 1].
    """
    import numpy as np
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)
