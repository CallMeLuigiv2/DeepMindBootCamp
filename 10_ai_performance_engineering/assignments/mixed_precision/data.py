"""Data loading for mixed precision and quantization experiments."""

from torch.utils.data import DataLoader

from shared_utils.data import load_cifar10


def load_cifar10_standard(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 with standard settings.

    Args:
        batch_size: Batch size.
        val_split: Fraction for validation.
        num_workers: Number of workers.
        augment: Whether to augment training data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    return load_cifar10(
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        augment=augment,
    )
