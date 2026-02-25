"""Model variants for profiling experiments.

Provides both a standard ResNet-18 and a deliberately inefficient variant
with anti-patterns to profile and fix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18Standard(nn.Module):
    """Standard ResNet-18 for CIFAR-10 (efficient implementation).

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # YOUR CODE HERE
        # Build efficient ResNet-18 for CIFAR-10:
        # conv1(3, 64, 3, 1, 1) -> BN -> ReLU -> 4 residual layers -> avgpool -> fc
        # Use in-place ReLU where possible
        raise NotImplementedError("Initialize ResNet18Standard")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 32, 32).

        Returns:
            Logits (B, num_classes).
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement ResNet18Standard.forward")


class ResNet18Inefficient(nn.Module):
    """Deliberately inefficient ResNet-18 with performance anti-patterns.

    Anti-patterns:
    - No in-place operations (creates unnecessary tensor copies)
    - Functional ReLU instead of in-place nn.ReLU

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # YOUR CODE HERE
        # Build the same architecture as ResNet18Standard but WITHOUT
        # in-place operations. Use F.relu (not inplace) everywhere.
        raise NotImplementedError("Initialize ResNet18Inefficient")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deliberate inefficiencies.

        Anti-patterns to include:
        - Use F.relu(x) instead of F.relu(x, inplace=True)
        - Clone tensors unnecessarily before residual connections

        Args:
            x: Input tensor (B, 3, 32, 32).

        Returns:
            Logits (B, num_classes).
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement ResNet18Inefficient.forward")
