"""
Fine-Tune Pretrained Transformer - Data Loading

Handles dataset loading, tokenization, and DataLoader creation for both
BERT fine-tuning and from-scratch training.
"""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer


class TokenizedDataset(Dataset):
    """Wraps a HuggingFace dataset with pre-tokenized inputs."""

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_and_tokenize_data(
    dataset_name: str = "glue",
    dataset_config: str = "sst2",
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    max_train_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset, BertTokenizer]:
    """Load and tokenize a classification dataset.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., 'sst2')
        model_name: Pretrained model name for tokenizer
        max_length: Maximum sequence length
        max_train_samples: Limit training samples (for data efficiency experiments)

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    # Load dataset
    if dataset_config:
        raw = load_dataset(dataset_name, dataset_config)
    else:
        raw = load_dataset(dataset_name)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Get text and label column names
    if dataset_name == "glue" and dataset_config == "sst2":
        text_col = "sentence"
        label_col = "label"
        train_split = "train"
        val_split = "validation"
    elif dataset_name == "imdb":
        text_col = "text"
        label_col = "label"
        train_split = "train"
        val_split = "test"
    else:
        # Default: try common column names
        cols = raw[list(raw.keys())[0]].column_names
        text_col = "text" if "text" in cols else "sentence"
        label_col = "label"
        train_split = "train"
        val_split = "validation" if "validation" in raw else "test"

    # Tokenize
    train_texts = raw[train_split][text_col]
    train_labels = raw[train_split][label_col]
    val_texts = raw[val_split][text_col]
    val_labels = raw[val_split][label_col]

    # Limit training samples if specified
    if max_train_samples is not None and max_train_samples > 0:
        train_texts = train_texts[:max_train_samples]
        train_labels = train_labels[:max_train_samples]

    train_encodings = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors=None,
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors=None,
    )

    train_dataset = TokenizedDataset(train_encodings, train_labels)
    val_dataset = TokenizedDataset(val_encodings, val_labels)

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(
    dataset_name: str = "glue",
    dataset_config: str = "sst2",
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 16,
    max_train_samples: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, BertTokenizer]:
    """Create DataLoaders for training and validation.

    Returns:
        train_loader, val_loader, tokenizer
    """
    train_dataset, val_dataset, tokenizer = load_and_tokenize_data(
        dataset_name, dataset_config, model_name, max_length, max_train_samples,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, tokenizer
