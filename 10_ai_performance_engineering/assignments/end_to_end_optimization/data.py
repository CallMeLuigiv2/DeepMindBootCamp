"""Optimized data pipeline for text classification.

Supports pre-tokenization, prefetching, and pinned memory.
Uses HuggingFace datasets for AG News / IMDB / SST-2.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TextClassificationDataset(Dataset):
    """Pre-tokenized text classification dataset.

    Args:
        input_ids: Tokenized input tensor (N, seq_len).
        labels: Label tensor (N,).
    """

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


def build_simple_tokenizer(texts: list[str], vocab_size: int = 30000) -> dict[str, int]:
    """Build a simple word-level tokenizer from texts.

    Args:
        texts: List of text strings.
        vocab_size: Maximum vocabulary size.

    Returns:
        Dictionary mapping tokens to indices.
    """
    from collections import Counter

    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())

    # Special tokens
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    for word, _ in word_counts.most_common(vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def tokenize_texts(
    texts: list[str],
    vocab: dict[str, int],
    max_length: int = 256,
) -> torch.Tensor:
    """Tokenize a list of texts into padded tensor.

    Args:
        texts: List of text strings.
        vocab: Token-to-index mapping.
        max_length: Maximum sequence length (pad/truncate to this).

    Returns:
        Tensor of shape (N, max_length) with token indices.
    """
    unk_idx = vocab.get("<unk>", 1)
    pad_idx = vocab.get("<pad>", 0)

    all_ids = []
    for text in texts:
        tokens = text.lower().split()[:max_length]
        ids = [vocab.get(t, unk_idx) for t in tokens]
        # Pad to max_length
        ids = ids + [pad_idx] * (max_length - len(ids))
        all_ids.append(ids)

    return torch.tensor(all_ids, dtype=torch.long)


def load_ag_news(
    batch_size: int = 16,
    max_seq_length: int = 256,
    vocab_size: int = 30000,
    pre_tokenize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load AG News dataset with optional pre-tokenization.

    Args:
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        vocab_size: Vocabulary size.
        pre_tokenize: If True, tokenize all text upfront and store as tensors.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        val_split: Fraction for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("ag_news")
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Build tokenizer from training data
    vocab = build_simple_tokenizer(train_texts, vocab_size=vocab_size)

    if pre_tokenize:
        # Tokenize everything upfront (fast: no per-batch tokenization)
        train_ids = tokenize_texts(train_texts, vocab, max_seq_length)
        test_ids = tokenize_texts(test_texts, vocab, max_seq_length)
        train_labels_t = torch.tensor(train_labels, dtype=torch.long)
        test_labels_t = torch.tensor(test_labels, dtype=torch.long)

        full_train = TextClassificationDataset(train_ids, train_labels_t)
        test_dataset = TextClassificationDataset(test_ids, test_labels_t)
    else:
        # Tokenize on-the-fly in __getitem__ (slow baseline)
        class OnTheFlyDataset(Dataset):
            def __init__(self, texts, labels, vocab, max_length):
                self.texts = texts
                self.labels = labels
                self.vocab = vocab
                self.max_length = max_length

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                text = self.texts[idx]
                tokens = text.lower().split()[:self.max_length]
                ids = [self.vocab.get(t, 1) for t in tokens]
                ids = ids + [0] * (self.max_length - len(ids))
                return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

        full_train = OnTheFlyDataset(train_texts, train_labels, vocab, max_seq_length)
        test_dataset = OnTheFlyDataset(test_texts, test_labels, vocab, max_seq_length)

    # Split train into train/val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    worker_kwargs = {}
    if persistent_workers and num_workers > 0:
        worker_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
