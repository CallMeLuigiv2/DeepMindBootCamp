"""Dataset loading framework for paper reproduction.

Provides standard dataset loading that matches the paper's data setup.
Modify augmentation and preprocessing to match your chosen paper.
"""

from torch.utils.data import DataLoader

from shared_utils.data import load_cifar10, load_mnist


def load_paper_dataset(
    paper: str = "resnet",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load the dataset for the chosen paper.

    Args:
        paper: Which paper ('resnet', 'batchnorm', 'ddpm', 'lora', 'vit').
        batch_size: Batch size.
        val_split: Validation split fraction.
        num_workers: Number of data loading workers.
        augment: Whether to augment training data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if paper in ("resnet", "batchnorm", "vit"):
        # ResNet paper: CIFAR-10 with standard augmentation
        # "4 pixels are padded on each side, and a 32x32 crop is randomly sampled"
        # "horizontal flip" (Section 4.2)
        return load_cifar10(
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            augment=augment,
        )
    elif paper == "ddpm":
        # DDPM: start with MNIST for feasibility, then CIFAR-10
        return load_mnist(
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
        )
    elif paper == "lora":
        # LoRA: need a text classification dataset
        # Use shared_utils or load from HuggingFace
        return load_cifar10(  # Placeholder -- replace with text data
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown paper: {paper}")
