"""Paper architecture stub.

Implement the core architecture of the paper you chose to reproduce.
Do NOT copy the authors' code -- build it from your understanding of the paper.

Select one and implement:
- Option A: ResNet-18/34 for CIFAR-10
- Option B: Network with/without Batch Normalization
- Option C: DDPM (U-Net + diffusion process)
- Option D: LoRA (low-rank adaptation layers)
- Option E: Vision Transformer (ViT) for CIFAR-10
"""

import torch
import torch.nn as nn


# ============================================================
# Option A: ResNet (He et al., 2015)
# ============================================================

class ResNetBlock(nn.Module):
    """Basic residual block for ResNet.

    Implement based on the paper's description:
    - Two 3x3 convolutions with batch normalization and ReLU
    - Skip connection (identity or projection shortcut)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for first convolution (2 for downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # YOUR CODE HERE
        # Read Section 3.2 of the paper carefully for the block structure.
        # Decision to document: projection shortcut vs zero-padding shortcut
        raise NotImplementedError("Implement ResNetBlock from paper description")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement ResNetBlock.forward")


class ResNetCIFAR(nn.Module):
    """ResNet adapted for CIFAR-10 (32x32 images).

    The paper trains on ImageNet (224x224). For CIFAR-10:
    - Use 3x3 conv with stride 1 (not 7x7 stride 2)
    - No max pooling after first conv
    - Document these deviations in your decision log

    Args:
        num_layers: Number of residual layers per stage (e.g., 3 for ResNet-20).
        num_classes: Number of output classes.
    """

    def __init__(self, num_layers: int = 3, num_classes: int = 10):
        super().__init__()
        # YOUR CODE HERE
        # Paper specifies: conv(3x3, 16 filters) -> 3 stages of 2n layers each
        # Stage 1: 16 filters, Stage 2: 32 filters (stride 2), Stage 3: 64 filters (stride 2)
        # For ResNet-20: n=3, For ResNet-32: n=5, For ResNet-56: n=9
        raise NotImplementedError("Implement ResNetCIFAR from paper")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement ResNetCIFAR.forward")


# ============================================================
# Plain network (baseline for comparison with ResNet)
# ============================================================

class PlainNetCIFAR(nn.Module):
    """Plain network (no skip connections) for comparison with ResNet.

    Same architecture as ResNetCIFAR but without residual connections.
    Used to demonstrate the degradation problem.

    Args:
        num_layers: Number of layers per stage.
        num_classes: Number of output classes.
    """

    def __init__(self, num_layers: int = 3, num_classes: int = 10):
        super().__init__()
        # YOUR CODE HERE
        # Same structure as ResNet but without skip connections
        raise NotImplementedError("Implement PlainNetCIFAR")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError("Implement PlainNetCIFAR.forward")
