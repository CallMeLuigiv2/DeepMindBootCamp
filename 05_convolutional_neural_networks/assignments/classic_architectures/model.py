"""Landmark CNN architectures: LeNet-5, VGG-11, ResNet-18, and PlainNet-18.

Implement each architecture as a PyTorch nn.Module. Count parameters manually
and verify against code. Do NOT use torchvision.models -- build from scratch.

Architectures:
    Part 1: LeNet-5 (LeCun et al., 1998) on MNIST, adapted for CIFAR-10
    Part 2: VGG-11 (Simonyan & Zisserman, 2015) on CIFAR-10
    Part 3: ResNet-18 (He et al., 2016) on CIFAR-10
    Part 5: PlainNet-18 (ResNet-18 without skip connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: LeNet-5
# ============================================================

class LeNet5(nn.Module):
    """LeNet-5 architecture (modernized with ReLU).

    Original paper: LeCun et al., 1998.

    For MNIST (Part 1):
        Input: (1, 32, 32) -- pad 28x28 MNIST images to 32x32
    For CIFAR-10 comparison (Part 4):
        Input: (3, 32, 32) -- change in_channels to 3

    Architecture:
        Conv2d(in_ch, 6, 5)  -> ReLU -> AvgPool(2)    -> (6, 14, 14)
        Conv2d(6, 16, 5)     -> ReLU -> AvgPool(2)     -> (16, 5, 5)
        Flatten                                         -> (400,)
        Linear(400, 120)     -> ReLU
        Linear(120, 84)      -> ReLU
        Linear(84, num_classes)

    Total parameters: ~62K (with in_channels=1)

    Args:
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR).
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        # YOUR CODE HERE: define layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, 32, 32).

        Returns:
            Logits of shape (B, num_classes).
        """
        # YOUR CODE HERE
        pass


# ============================================================
# Part 2: VGG-11
# ============================================================

class VGG11(nn.Module):
    """VGG-11 (configuration A) with BatchNorm, adapted for CIFAR-10.

    Original paper: Simonyan & Zisserman, 2015.
    Input: (3, 32, 32)

    Feature extractor:
        Block 1: Conv(3,64,3,p=1) -> BN -> ReLU -> MaxPool(2)      -> (64, 16, 16)
        Block 2: Conv(64,128,3,p=1) -> BN -> ReLU -> MaxPool(2)     -> (128, 8, 8)
        Block 3: Conv(128,256,3,p=1) -> BN -> ReLU
                 Conv(256,256,3,p=1) -> BN -> ReLU -> MaxPool(2)     -> (256, 4, 4)
        Block 4: Conv(256,512,3,p=1) -> BN -> ReLU
                 Conv(512,512,3,p=1) -> BN -> ReLU -> MaxPool(2)     -> (512, 2, 2)
        Block 5: Conv(512,512,3,p=1) -> BN -> ReLU
                 Conv(512,512,3,p=1) -> BN -> ReLU -> MaxPool(2)     -> (512, 1, 1)

    Classifier (GAP variant, ~9.2M total):
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(512, num_classes)

    Classifier (Full FC variant, ~28M total):
        Flatten -> Linear(512, 4096) -> ReLU -> Dropout(0.5)
               -> Linear(4096, 4096) -> ReLU -> Dropout(0.5)
               -> Linear(4096, num_classes)

    Args:
        num_classes: Number of output classes.
        use_gap: If True, use Global Average Pooling classifier (lighter).
                 If False, use full FC layers.
    """

    def __init__(self, num_classes: int = 10, use_gap: bool = True):
        super().__init__()
        # YOUR CODE HERE: define feature extractor blocks
        # self.features = nn.Sequential(...)

        # YOUR CODE HERE: define classifier (GAP or FC based on use_gap)
        # self.classifier = nn.Sequential(...)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits of shape (B, num_classes).
        """
        # YOUR CODE HERE
        pass


# ============================================================
# Part 3: ResNet-18
# ============================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet.

    Structure:
        x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+shortcut) -> ReLU

    The shortcut connection is the key innovation:
        - If dimensions match: identity shortcut (just add x)
        - If dimensions change: 1x1 conv shortcut to match dimensions

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first conv (2 for downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # YOUR CODE HERE: define conv1, bn1, conv2, bn2
        # Hint: use bias=False when followed by BatchNorm

        # YOUR CODE HERE: define shortcut
        # If stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #     )
        # else:
        #     self.shortcut = nn.Sequential()  # identity
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor (same spatial dims if stride=1, halved if stride=2).
        """
        # YOUR CODE HERE
        # identity = self.shortcut(x)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += identity       # <-- THE SKIP CONNECTION
        # out = F.relu(out)
        # return out
        pass


class PlainBlock(nn.Module):
    """Plain block WITHOUT skip connections (for the skip connection experiment).

    Same as BasicBlock but without the residual addition.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # YOUR CODE HERE: same convolutions as BasicBlock, but no shortcut
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass WITHOUT skip connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # YOUR CODE HERE
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = F.relu(out)     # <-- NO skip connection
        # return out
        pass


class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 input).

    Key difference from ImageNet ResNet-18: uses 3x3 stem with stride 1
    (not 7x7 stride 2), and no initial max pooling.

    Architecture:
        Stem: Conv(3, 64, 3, s=1, p=1) -> BN -> ReLU           -> (64, 32, 32)
        Layer 1: BasicBlock(64, 64)   x 2                        -> (64, 32, 32)
        Layer 2: BasicBlock(64, 128, s=2) + BasicBlock(128, 128) -> (128, 16, 16)
        Layer 3: BasicBlock(128, 256, s=2) + BasicBlock(256, 256)-> (256, 8, 8)
        Layer 4: BasicBlock(256, 512, s=2) + BasicBlock(512, 512)-> (512, 4, 4)
        AdaptiveAvgPool2d(1)                                      -> (512, 1, 1)
        Linear(512, num_classes)

    Total parameters: ~11.2M

    Args:
        block_class: Block class to use (BasicBlock for ResNet, PlainBlock for PlainNet).
        num_classes: Number of output classes.
    """

    def __init__(self, block_class=None, num_classes: int = 10):
        super().__init__()
        if block_class is None:
            block_class = BasicBlock

        # YOUR CODE HERE: define stem (conv1, bn1)
        # YOUR CODE HERE: define layer1, layer2, layer3, layer4
        # YOUR CODE HERE: define avgpool and fc

        # Hint for making layers:
        # def _make_layer(self, block_class, in_ch, out_ch, num_blocks, stride):
        #     layers = [block_class(in_ch, out_ch, stride)]
        #     for _ in range(1, num_blocks):
        #         layers.append(block_class(out_ch, out_ch, 1))
        #     return nn.Sequential(*layers)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).

        Returns:
            Logits of shape (B, num_classes).
        """
        # YOUR CODE HERE
        pass


def PlainNet18(num_classes: int = 10) -> ResNet18:
    """Create a PlainNet-18 (ResNet-18 without skip connections).

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNet18 instance using PlainBlock instead of BasicBlock.
    """
    return ResNet18(block_class=PlainBlock, num_classes=num_classes)


# ============================================================
# Factory Function
# ============================================================

def create_model(arch: str, config: dict = None) -> nn.Module:
    """Create a model by architecture name.

    Args:
        arch: Architecture name ('lenet5', 'vgg11', 'resnet18', 'plainnet18').
        config: Optional architecture-specific config dict.

    Returns:
        Instantiated model.
    """
    config = config or {}
    num_classes = config.get("num_classes", 10)

    if arch == "lenet5":
        return LeNet5(in_channels=config.get("in_channels", 3), num_classes=num_classes)
    elif arch == "vgg11":
        return VGG11(num_classes=num_classes, use_gap=config.get("use_gap", True))
    elif arch == "vgg11_fc":
        return VGG11(num_classes=num_classes, use_gap=False)
    elif arch == "resnet18":
        return ResNet18(block_class=BasicBlock, num_classes=num_classes)
    elif arch == "plainnet18":
        return PlainNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
