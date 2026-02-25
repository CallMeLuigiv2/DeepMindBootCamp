"""CNN architecture for CIFAR-10.

Implement a small CNN (<500K parameters) that achieves at least 85% test accuracy.

Architecture hint (modify and improve):
    Input: (B, 3, 32, 32)
    Block 1: Conv2d(3, 32, 3, padding=1) -> BN -> ReLU -> Conv2d(32, 32, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
      Output: (B, 32, 16, 16)
    Block 2: Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> Conv2d(64, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
      Output: (B, 64, 8, 8)
    Block 3: Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> Conv2d(128, 128, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
      Output: (B, 128, 4, 4)
    Flatten: (B, 2048)
    Dropout(0.3) -> Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10)
    Parameter count: approximately 400K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10Net(nn.Module):
    """Small CNN for CIFAR-10 classification.

    Requirements:
    - At least 3 conv layers with increasing channel counts (e.g., 32 -> 64 -> 128)
    - BatchNorm2d after each conv layer
    - MaxPool2d or strided convolutions for downsampling
    - Dropout for regularization
    - Fully connected classifier head
    - Total parameters under 500K

    Args:
        num_classes: Number of output classes (default: 10).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # YOUR CODE HERE: define the convolutional blocks
        # Hint: use nn.Sequential for each block
        # self.block1 = nn.Sequential(...)
        # self.block2 = nn.Sequential(...)
        # self.block3 = nn.Sequential(...)

        # YOUR CODE HERE: define the classifier head
        # self.classifier = nn.Sequential(...)

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # YOUR CODE HERE
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> CIFAR10Net:
    """Create a model from config.

    Args:
        config: Model configuration dictionary.

    Returns:
        Instantiated CIFAR10Net model.
    """
    return CIFAR10Net(num_classes=config.get("num_classes", 10))
