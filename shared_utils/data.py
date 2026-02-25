"""Data loading utilities for common datasets used across assignments."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(
    batch_size: int = 64,
    val_split: float = 0.1,
    data_dir: Optional[str] = None,
    flatten: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST dataset with train/val/test splits.

    Args:
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of training data to use for validation.
        data_dir: Directory to download/cache data. Defaults to project root/data.
        flatten: If True, flatten images to (batch_size, 784).
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = data_dir or str(DATA_DIR)
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    full_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def load_cifar10(
    batch_size: int = 64,
    val_split: float = 0.1,
    data_dir: Optional[str] = None,
    augment: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 dataset with train/val/test splits and optional augmentation.

    Args:
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of training data to use for validation.
        data_dir: Directory to download/cache data. Defaults to project root/data.
        augment: If True, apply standard data augmentation to training set.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = data_dir or str(DATA_DIR)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def load_text_corpus(
    path: str,
    seq_length: int = 128,
    batch_size: int = 32,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Load a text file as a character-level dataset for language modeling.

    Args:
        path: Path to the text file.
        seq_length: Length of each input sequence.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader, char_to_idx, idx_to_char).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    # Create sequences
    n_sequences = (len(encoded) - 1) // seq_length
    inputs = encoded[: n_sequences * seq_length].view(n_sequences, seq_length)
    targets = encoded[1: n_sequences * seq_length + 1].view(n_sequences, seq_length)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, char_to_idx, idx_to_char


def load_shakespeare(
    seq_length: int = 128,
    batch_size: int = 32,
    val_split: float = 0.1,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Download and load the Tiny Shakespeare dataset for language modeling.

    Args:
        seq_length: Length of each input sequence.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of data to use for validation.
        data_dir: Directory to cache the downloaded file.

    Returns:
        Tuple of (train_loader, val_loader, char_to_idx, idx_to_char).
    """
    import urllib.request

    root = Path(data_dir or str(DATA_DIR))
    root.mkdir(parents=True, exist_ok=True)
    filepath = root / "tiny_shakespeare.txt"

    if not filepath.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)

    return load_text_corpus(str(filepath), seq_length, batch_size, val_split)
