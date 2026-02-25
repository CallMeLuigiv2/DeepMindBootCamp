"""Model compatible with DDP, gradient checkpointing, and torch.compile.

Implements a ResNet-18 variant for CIFAR-10 that supports all advanced
training techniques from this assignment.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class BasicBlock(nn.Module):
    """Basic residual block for ResNet.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # YOUR CODE HERE
        # Build the residual block:
        # 1. conv1: Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        # 2. bn1: BatchNorm2d(out_channels)
        # 3. relu: ReLU(inplace=True)
        # 4. conv2: Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        # 5. bn2: BatchNorm2d(out_channels)
        # 6. shortcut: if stride != 1 or in_channels != out_channels, use
        #    Sequential(Conv2d(in_channels, out_channels, 1, stride, bias=False), BatchNorm2d)
        #    else Identity
        raise NotImplementedError("Initialize BasicBlock")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor.
        """
        # YOUR CODE HERE
        # 1. identity = shortcut(x)
        # 2. out = relu(bn1(conv1(x)))
        # 3. out = bn2(conv2(out))
        # 4. out = relu(out + identity)
        raise NotImplementedError("Implement BasicBlock.forward")


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 images).

    Modifications from standard ResNet-18:
    - First conv: 3x3, stride 1, padding 1 (instead of 7x7, stride 2)
    - No max pool after first conv
    - Supports gradient checkpointing

    Args:
        num_classes: Number of output classes.
        use_checkpointing: Whether to use gradient checkpointing for layers.
    """

    def __init__(self, num_classes: int = 10, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # YOUR CODE HERE
        # Build the ResNet-18 for CIFAR-10:
        # 1. conv1: Conv2d(3, 64, 3, 1, 1, bias=False)
        # 2. bn1: BatchNorm2d(64)
        # 3. relu: ReLU(inplace=True)
        # 4. layer1: _make_layer(64, 64, 2, stride=1)
        # 5. layer2: _make_layer(64, 128, 2, stride=2)
        # 6. layer3: _make_layer(128, 256, 2, stride=2)
        # 7. layer4: _make_layer(256, 512, 2, stride=2)
        # 8. avgpool: AdaptiveAvgPool2d((1, 1))
        # 9. fc: Linear(512, num_classes)
        raise NotImplementedError("Initialize ResNet18CIFAR")

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a residual layer with the given number of blocks.

        Args:
            in_channels: Input channels for the first block.
            out_channels: Output channels.
            num_blocks: Number of BasicBlocks.
            stride: Stride for the first block.

        Returns:
            Sequential module containing all blocks.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _make_layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits of shape (B, num_classes).
        """
        # YOUR CODE HERE
        # 1. x = relu(bn1(conv1(x)))
        # 2. For each layer (layer1-layer4):
        #    if self.use_checkpointing and self.training:
        #        x = checkpoint(layer, x, use_reentrant=False)
        #    else:
        #        x = layer(x)
        # 3. x = avgpool(x).flatten(1)
        # 4. x = fc(x)
        raise NotImplementedError("Implement ResNet18CIFAR.forward")
