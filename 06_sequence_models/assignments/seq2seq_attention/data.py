"""
Seq2Seq with Attention - Data Generation and Processing

Generates synthetic date format conversion pairs and provides
character-level tokenization and DataLoader creation.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Date formats for synthetic data generation
FORMATS = [
    "%B %d, %Y",   # January 05, 2023
    "%d %B %Y",    # 05 January 2023
    "%b %d, %Y",   # Jan 05, 2023
    "%d %b %Y",    # 05 Jan 2023
    "%B %d %Y",    # January 05 2023
    "%d/%m/%Y",    # 05/01/2023
    "%m/%d/%Y",    # 01/05/2023
]

# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


def generate_date_pair() -> Tuple[str, str]:
    """Generate a random (human_date, machine_date) pair.

    Returns:
        Tuple of (human-readable date string, ISO format date string)
    """
    start = datetime(1950, 1, 1)
    end = datetime(2030, 12, 31)
    delta = end - start
    random_date = start + timedelta(days=random.randint(0, delta.days))

    human_format = random.choice(FORMATS)
    human_date = random_date.strftime(human_format)
    machine_date = random_date.strftime("%Y-%m-%d")

    return human_date, machine_date


def generate_dataset(
    num_samples: int = 50000, seed: int = 42
) -> List[Tuple[str, str]]:
    """Generate the full synthetic date dataset.

    Args:
        num_samples: Total number of date pairs to generate
        seed: Random seed for reproducibility

    Returns:
        List of (human_date, machine_date) tuples
    """
    random.seed(seed)
    return [generate_date_pair() for _ in range(num_samples)]


def build_vocabulary(
    pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Build character-level vocabularies for source and target.

    Args:
        pairs: List of (source_string, target_string) pairs

    Returns:
        src_char2idx: Source character to index mapping
        trg_char2idx: Target character to index mapping
        src_idx2char: Source index to character mapping
        trg_idx2char: Target index to character mapping
    """
    src_chars = set()
    trg_chars = set()

    for src, trg in pairs:
        src_chars.update(src)
        trg_chars.update(trg)

    # Build mappings with special tokens at the front
    special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

    src_char2idx = {tok: i for i, tok in enumerate(special_tokens)}
    for i, ch in enumerate(sorted(src_chars), start=len(special_tokens)):
        src_char2idx[ch] = i

    trg_char2idx = {tok: i for i, tok in enumerate(special_tokens)}
    for i, ch in enumerate(sorted(trg_chars), start=len(special_tokens)):
        trg_char2idx[ch] = i

    src_idx2char = {i: ch for ch, i in src_char2idx.items()}
    trg_idx2char = {i: ch for ch, i in trg_char2idx.items()}

    return src_char2idx, trg_char2idx, src_idx2char, trg_idx2char


def encode_sequence(text: str, char2idx: Dict[str, int]) -> List[int]:
    """Convert a string to a list of token indices (with BOS and EOS)."""
    return [BOS_IDX] + [char2idx[ch] for ch in text] + [EOS_IDX]


class DateDataset(Dataset):
    """Dataset for date format conversion pairs.

    Each item is a tuple of (source_indices, target_indices) tensors.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_char2idx: Dict[str, int],
        trg_char2idx: Dict[str, int],
    ):
        self.pairs = pairs
        self.src_char2idx = src_char2idx
        self.trg_char2idx = trg_char2idx

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, trg_text = self.pairs[idx]
        src_ids = encode_sequence(src_text, self.src_char2idx)
        trg_ids = encode_sequence(trg_text, self.trg_char2idx)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that pads sequences to the same length.

    Returns:
        src_padded: (batch, max_src_len) - Padded source sequences
        trg_padded: (batch, max_trg_len) - Padded target sequences
        src_mask: (batch, max_src_len) - 1 for real tokens, 0 for padding
    """
    src_seqs, trg_seqs = zip(*batch)

    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_seqs, batch_first=True, padding_value=PAD_IDX)
    src_mask = (src_padded != PAD_IDX).float()

    return src_padded, trg_padded, src_mask


def create_dataloaders(
    num_samples: int = 50000,
    train_size: int = 40000,
    val_size: int = 5000,
    test_size: int = 5000,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict, Dict, Dict]:
    """Generate data and create train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader,
        src_char2idx, trg_char2idx, src_idx2char, trg_idx2char
    """
    pairs = generate_dataset(num_samples, seed)

    # Split
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size : train_size + val_size]
    test_pairs = pairs[train_size + val_size : train_size + val_size + test_size]

    # Build vocabularies from training data
    src_char2idx, trg_char2idx, src_idx2char, trg_idx2char = build_vocabulary(train_pairs)

    # Create datasets
    train_ds = DateDataset(train_pairs, src_char2idx, trg_char2idx)
    val_ds = DateDataset(val_pairs, src_char2idx, trg_char2idx)
    test_ds = DateDataset(test_pairs, src_char2idx, trg_char2idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return (
        train_loader, val_loader, test_loader,
        src_char2idx, trg_char2idx, src_idx2char, trg_idx2char,
    )
