"""Data loading for custom autograd experiments.

- MNIST for binary activation network (STE experiments)
- Synthetic regression data for asymmetric loss experiments
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from shared_utils.data import load_mnist


def load_mnist_flat(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST with flattened images for the binary activation network.

    Args:
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of training data for validation.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    return load_mnist(
        batch_size=batch_size,
        val_split=val_split,
        flatten=True,
        num_workers=num_workers,
    )


def generate_regression_data(
    n_train: int = 1000,
    n_test: int = 200,
    noise_std: float = 0.3,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Generate synthetic regression data where underestimation is costly.

    Simulates a resource usage prediction task: the true function is a noisy
    sinusoid, and we want a model that slightly overestimates to avoid outages.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of additive noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    torch.manual_seed(seed)

    # True function: f(x) = 2*sin(x) + 0.5*x + 3 (resource usage pattern)
    def true_fn(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sin(x) + 0.5 * x + 3.0

    # Training data
    x_train = torch.rand(n_train, 1) * 8.0 - 2.0  # x in [-2, 6]
    y_train = true_fn(x_train) + torch.randn(n_train, 1) * noise_std

    # Test data
    x_test = torch.linspace(-2.0, 6.0, n_test).unsqueeze(1)
    y_test = true_fn(x_test)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=64,
        shuffle=False,
    )

    return train_loader, test_loader
