"""Model with standard layers, compatible with mixed precision and quantization.

Implements ResNet-18 for CIFAR-10 with clean layer structure suitable for
both mixed precision training and post-training quantization.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for first convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # YOUR CODE HERE
        # Build residual block:
        # conv1 -> bn1 -> relu -> conv2 -> bn2
        # shortcut (if dimensions change)
        # Use nn.ReLU(inplace=True) for standard (non-quantized) use
        raise NotImplementedError("Initialize BasicBlock")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement BasicBlock.forward")


class ResNet18ForQuantization(nn.Module):
    """ResNet-18 for CIFAR-10 with quantization support.

    When preparing for quantization, QuantStub/DeQuantStub can be added
    as wrappers around the model.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # YOUR CODE HERE
        # Build ResNet-18 for CIFAR-10 (32x32 input):
        # conv1(3x3, stride 1) -> bn1 -> relu
        # layer1 (2 blocks, 64 ch) -> layer2 (2 blocks, 128 ch, stride 2)
        # layer3 (2 blocks, 256 ch, stride 2) -> layer4 (2 blocks, 512 ch, stride 2)
        # avgpool -> fc
        raise NotImplementedError("Initialize ResNet18ForQuantization")

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        # YOUR CODE HERE
        raise NotImplementedError("Implement _make_layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, 3, 32, 32).

        Returns:
            Logits (B, num_classes).
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement ResNet18ForQuantization.forward")
