"""Common utilities used across all assignments."""

import os
import random
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of model architecture and parameter counts."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"Model: {model.__class__.__name__}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Non-trainable:        {total - trainable:,}")
    print()
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({params:,} params)")


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Usage:
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(max_epochs):
            train_loss = train(...)
            val_loss = validate(...)
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **extra,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **extra,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class TrainingLogger:
    """Simple logger that records metrics per epoch and saves to JSON."""

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def log(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(value)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.history = json.load(f)
