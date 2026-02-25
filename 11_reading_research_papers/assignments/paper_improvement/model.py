"""Base model + improvement stubs.

Uses a configuration flag to switch between the original method and
your improvement. Both variants share the same base architecture so
comparisons are fair.

Example improvements (for ResNet):
- Replace ReLU with GELU/SiLU
- Add Squeeze-and-Excitation blocks
- Replace BatchNorm with GroupNorm/LayerNorm
- Add stochastic depth (randomly drop residual blocks)
- Pre-activation residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseResNetBlock(nn.Module):
    """Standard residual block (baseline, from Assignment 2).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # YOUR CODE HERE
        # Copy your working implementation from reproduce_paper/model.py
        raise NotImplementedError("Copy baseline block from Assignment 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Copy baseline forward from Assignment 2")


class ImprovedResNetBlock(nn.Module):
    """Improved residual block with your modification.

    MODIFICATION: [Describe your specific change here]

    Hypothesis: [Why you expect this to help]

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for downsampling.
        improvement_config: Dictionary with improvement-specific parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        improvement_config: dict = None,
    ):
        super().__init__()
        if improvement_config is None:
            improvement_config = {}

        # YOUR CODE HERE
        # Implement the improved block.
        # Keep the same overall structure but add your modification.
        # Examples:
        # - Add SE block after second conv
        # - Replace ReLU with GELU
        # - Add stochastic depth (random block dropping)
        raise NotImplementedError("Implement your improved residual block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement improved forward")


class PaperModel(nn.Module):
    """Model with config flag to switch between baseline and improved variants.

    Args:
        num_layers: Number of residual layers per stage.
        num_classes: Number of output classes.
        use_improvement: If True, use ImprovedResNetBlock; else BaseResNetBlock.
        improvement_config: Configuration for the improvement.
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_classes: int = 10,
        use_improvement: bool = False,
        improvement_config: dict = None,
    ):
        super().__init__()
        self.use_improvement = use_improvement

        block_class = ImprovedResNetBlock if use_improvement else BaseResNetBlock
        block_kwargs = {"improvement_config": improvement_config} if use_improvement else {}

        # YOUR CODE HERE
        # Build the full model using the selected block class:
        # - conv1(3, 16, 3, 1, 1) -> bn1 -> relu
        # - stage1: num_layers blocks, 16 channels
        # - stage2: num_layers blocks, 32 channels, stride=2
        # - stage3: num_layers blocks, 64 channels, stride=2
        # - avgpool -> fc(64, num_classes)
        raise NotImplementedError("Initialize PaperModel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement PaperModel.forward")
