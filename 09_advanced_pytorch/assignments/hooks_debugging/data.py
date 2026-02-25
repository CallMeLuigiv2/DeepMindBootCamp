"""Data loading for hooks and debugging experiments.

- Standard MNIST for gradient flow and pruning experiments
- Domain adaptation split: noisy source / clean target for DANN
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from shared_utils.data import load_mnist


def load_mnist_flat(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST with flattened images.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    return load_mnist(
        batch_size=batch_size,
        val_split=val_split,
        flatten=True,
        num_workers=num_workers,
    )


def load_dann_data(
    batch_size: int = 128,
    noise_std: float = 0.5,
    digits: list[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST domain adaptation data (digits 0-4 only).

    Source domain: digits with additive Gaussian noise.
    Target domain: clean digits.

    Args:
        batch_size: Batch size.
        noise_std: Standard deviation of Gaussian noise for source domain.
        digits: Which digits to include (default: [0, 1, 2, 3, 4]).
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (source_loader, target_train_loader, target_test_loader).
        Source loader yields (noisy_images, labels, domain_label=0).
        Target loaders yield (clean_images, labels, domain_label=1).
    """
    if digits is None:
        digits = [0, 1, 2, 3, 4]

    import torchvision
    import torchvision.transforms as transforms
    from pathlib import Path

    data_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    full_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Filter to selected digits
    train_indices = [i for i, (_, label) in enumerate(full_train) if label in digits]
    test_indices = [i for i, (_, label) in enumerate(full_test) if label in digits]

    # Collect filtered data
    train_images = torch.stack([full_train[i][0] for i in train_indices])
    train_labels = torch.tensor([full_train[i][1] for i in train_indices])
    test_images = torch.stack([full_test[i][0] for i in test_indices])
    test_labels = torch.tensor([full_test[i][1] for i in test_indices])

    # Source: noisy version of training data
    source_images = train_images + torch.randn_like(train_images) * noise_std
    source_domain = torch.zeros(len(train_images), dtype=torch.long)

    # Target: clean training and test data
    target_domain = torch.ones(len(train_images), dtype=torch.long)

    source_loader = DataLoader(
        TensorDataset(source_images, train_labels, source_domain),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    target_train_loader = DataLoader(
        TensorDataset(train_images, train_labels, target_domain),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    target_test_loader = DataLoader(
        TensorDataset(test_images, test_labels, torch.ones(len(test_images), dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return source_loader, target_train_loader, target_test_loader
