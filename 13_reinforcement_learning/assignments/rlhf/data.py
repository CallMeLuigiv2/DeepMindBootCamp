"""
RLHF Datasets
===============

Provides dataset infrastructure for RLHF training:
- PreferenceDataset: preference pairs for reward model and DPO training (pre-written)
- PromptDataset: prompts for PPO fine-tuning (pre-written)
- Preference data generation utilities
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Preference Dataset (pre-written)
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """Dataset of preference pairs for reward model and DPO training.

    Each example contains a prompt, a chosen (preferred) response, and a
    rejected (dispreferred) response.

    Args:
        data: List of dicts with keys "prompt", "chosen", "rejected".
        tokenizer: Tokenizer for encoding text.
        max_length: Maximum sequence length after tokenization.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized chosen and rejected sequences.

        Returns:
            Dictionary with:
            - chosen_ids: token IDs for prompt + chosen response
            - chosen_mask: attention mask for chosen
            - rejected_ids: token IDs for prompt + rejected response
            - rejected_mask: attention mask for rejected
            - prompt: raw prompt text
        """
        item = self.data[idx]
        prompt = item["prompt"]
        chosen_text = prompt + item["chosen"]
        rejected_text = prompt + item["rejected"]

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt": prompt,
        }

    @staticmethod
    def load_from_json(path: str, tokenizer, max_length: int = 512) -> "PreferenceDataset":
        """Load a preference dataset from a JSON file.

        Args:
            path: Path to the JSON file.
            tokenizer: Tokenizer for encoding.
            max_length: Maximum sequence length.

        Returns:
            PreferenceDataset instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return PreferenceDataset(data, tokenizer, max_length)

    @staticmethod
    def save_to_json(data: List[Dict[str, str]], path: str) -> None:
        """Save preference data to a JSON file.

        Args:
            data: List of preference dicts.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def train_val_split(
        self, val_fraction: float = 0.2
    ) -> tuple:
        """Split into train and validation datasets.

        Args:
            val_fraction: Fraction of data for validation.

        Returns:
            (train_dataset, val_dataset) tuple.
        """
        n_val = int(len(self.data) * val_fraction)
        indices = list(range(len(self.data)))
        random.shuffle(indices)

        val_data = [self.data[i] for i in indices[:n_val]]
        train_data = [self.data[i] for i in indices[n_val:]]

        return (
            PreferenceDataset(train_data, self.tokenizer, self.max_length),
            PreferenceDataset(val_data, self.tokenizer, self.max_length),
        )


# ---------------------------------------------------------------------------
# Prompt Dataset (pre-written)
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """Dataset of prompts for PPO fine-tuning.

    Provides tokenized prompts that the policy model will generate
    completions for during RLHF training.

    Args:
        prompts: List of prompt strings.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum prompt length.
    """

    def __init__(
        self,
        prompts: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized prompt.

        Returns:
            Dictionary with:
            - input_ids: token IDs for the prompt
            - attention_mask: attention mask
            - prompt: raw prompt text
        """
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": prompt,
        }


# ---------------------------------------------------------------------------
# Default prompts (pre-written)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "The best way to learn programming is",
    "In a healthy relationship, it is important to",
    "The meaning of life is",
    "When faced with a difficult problem, you should",
    "A good leader always",
]

TRAINING_PROMPTS = [
    "The most important thing about education is",
    "Technology has changed the world by",
    "A balanced diet should include",
    "The key to success in any field is",
    "Climate change can be addressed by",
    "The purpose of art is",
    "Good communication requires",
    "In times of uncertainty, people should",
    "The internet has transformed",
    "A healthy work-life balance means",
    "The greatest scientific discovery was",
    "Democracy works best when",
    "The future of artificial intelligence will",
    "Reading books is important because",
    "The secret to happiness is",
    "Mental health awareness matters because",
    "Travel broadens the mind by",
    "Exercise is beneficial because",
    "A strong community is built on",
    "The role of music in society is",
    "Education should focus on",
    "Environmental protection requires",
    "Innovation happens when",
    "The value of friendship is",
    "Good parenting involves",
    "History teaches us that",
    "Space exploration is worthwhile because",
    "Effective teamwork requires",
    "The importance of sleep is",
    "Financial literacy means understanding",
    "Cultural diversity enriches society by",
    "The scientific method works because",
    "A meaningful career involves",
    "Social media has impacted",
    "The power of storytelling lies in",
    "Sustainable living means",
    "Critical thinking is essential for",
    "The beauty of mathematics is",
    "Volunteering benefits",
    "The challenge of modern life is",
    "Learning a new language helps",
    "Creativity can be fostered by",
    "The ethics of technology involve",
    "Personal growth requires",
    "The importance of voting is",
    "Healthy aging involves",
    "The ocean is important because",
    "Effective leadership means",
    "The value of patience is",
    "Progress in society comes from",
]
