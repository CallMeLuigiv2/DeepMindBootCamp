"""
Transformer from Scratch - Data Loading

Provides datasets for:
1. Number sorting (encoder-decoder Transformer)
2. Character-level language modeling (GPT decoder-only)
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# Special tokens for sorting task
PAD_ID = 0
SOS_ID = 101
EOS_ID = 102


class SortingDataset(IterableDataset):
    """Infinite dataset that generates random sequences and their sorted versions.

    Generates pairs like:
        Input:  [SOS, 7, 3, 9, 1, 5, EOS]
        Target: [SOS, 1, 3, 5, 7, 9, EOS]

    Numbers are in range [1, 100] (0 is reserved for PAD).

    Args:
        seq_len_min: Minimum sequence length (number of integers)
        seq_len_max: Maximum sequence length
        seed: Random seed for reproducibility (None for no seed)
    """

    def __init__(self, seq_len_min: int = 5, seq_len_max: int = 10, seed: Optional[int] = None):
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            length = rng.randint(self.seq_len_min, self.seq_len_max)
            numbers = [rng.randint(1, 100) for _ in range(length)]
            sorted_numbers = sorted(numbers)

            src = [SOS_ID] + numbers + [EOS_ID]
            tgt = [SOS_ID] + sorted_numbers + [EOS_ID]

            yield torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def sorting_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate sorting examples with padding."""
    from torch.nn.utils.rnn import pad_sequence

    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_padded


class CharDataset(Dataset):
    """Character-level language modeling dataset.

    Splits text into overlapping windows of (context_length + 1) characters.
    Input: characters 0 to context_length-1
    Target: characters 1 to context_length

    Args:
        text: Raw text corpus
        context_length: Number of characters per training sample
    """

    def __init__(self, text: str, context_length: int = 128):
        self.context_length = context_length

        # Build character vocabulary
        chars = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)

        # Encode entire text
        self.data = torch.tensor(
            [self.char2idx[ch] for ch in text], dtype=torch.long
        )

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]  # Input: chars 0..context_length-1
        y = chunk[1:]   # Target: chars 1..context_length
        return x, y

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a string."""
        return "".join(self.idx2char.get(i, "?") for i in token_ids)


def load_text_corpus(path: str) -> str:
    """Load a text file for character-level language modeling.

    If the file doesn't exist, downloads a small Shakespeare corpus.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # Download Shakespeare if no corpus specified
    print(f"Corpus not found at {path}. Downloading Shakespeare...")
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_sorting_dataloader(
    batch_size: int = 64,
    seq_len_min: int = 5,
    seq_len_max: int = 10,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for the sorting task (infinite)."""
    dataset = SortingDataset(seq_len_min, seq_len_max, seed)
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=sorting_collate_fn,
    )


def create_char_dataloaders(
    corpus_path: str,
    context_length: int = 128,
    batch_size: int = 64,
    train_split: float = 0.9,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, CharDataset]:
    """Create train and validation DataLoaders for character LM.

    Returns:
        train_loader, val_loader, dataset (for vocab info)
    """
    text = load_text_corpus(corpus_path)
    split_idx = int(len(text) * train_split)

    train_dataset = CharDataset(text[:split_idx], context_length)
    val_dataset = CharDataset(text[split_idx:], context_length)

    # Share vocabulary (use train vocab)
    val_dataset.char2idx = train_dataset.char2idx
    val_dataset.idx2char = train_dataset.idx2char
    val_dataset.vocab_size = train_dataset.vocab_size
    val_dataset.data = torch.tensor(
        [train_dataset.char2idx.get(ch, 0) for ch in text[split_idx:]],
        dtype=torch.long,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, train_dataset
