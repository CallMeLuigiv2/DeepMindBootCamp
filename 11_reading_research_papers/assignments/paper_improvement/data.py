"""Dataset loading (same as reproduce_paper)."""

from torch.utils.data import DataLoader

from shared_utils.data import load_cifar10


def load_experiment_data(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 for A/B experiments.

    Uses identical data for both baseline and improved variants
    to ensure fair comparison.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    return load_cifar10(
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        augment=augment,
    )
