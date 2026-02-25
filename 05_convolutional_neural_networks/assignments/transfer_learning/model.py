"""Model definitions for the Transfer Learning project.

Provides factory functions for each transfer learning strategy:
- Baseline: small CNN from scratch
- Strategy 1: frozen backbone, new classification head
- Strategy 2: partial fine-tuning (last block + head)
- Strategy 3: full fine-tuning with differential learning rates

All strategies use ImageNet-pretrained backbones (except baseline).
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# Baseline: Small CNN from Scratch
# ============================================================

class SimpleCNN(nn.Module):
    """Simple CNN baseline trained from scratch.

    Used to quantify the benefit of transfer learning.
    4 conv blocks with increasing channels, GAP, and linear classifier.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 28x28 -> 1x1 (GAP)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# Strategy 1: Feature Extraction (Frozen Backbone)
# ============================================================

def create_frozen_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    """Create a model with a completely frozen pretrained backbone.

    Only the new classification head is trainable.

    Args:
        num_classes: Number of output classes.
        backbone: Pretrained model name ('resnet50' or 'efficientnet_b0').

    Returns:
        Model with frozen backbone and trainable head.
    """
    if backbone == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")

        # Freeze ALL backbone parameters
        for param in model.parameters():
            param.requires_grad = False

        # Replace classification head (these params are trainable by default)
        num_features = model.fc.in_features  # 2048
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


# ============================================================
# Strategy 2: Partial Fine-tuning
# ============================================================

def create_partial_finetune_model(
    num_classes: int,
    backbone: str = "resnet50",
    unfreeze_from: str = "layer4",
) -> nn.Module:
    """Create a model with partial fine-tuning.

    Freeze everything except the last residual block and the classification head.

    Args:
        num_classes: Number of output classes.
        backbone: Pretrained model name.
        unfreeze_from: Which layer to unfreeze from ('layer3', 'layer4').

    Returns:
        Model with partially frozen backbone.
    """
    # YOUR CODE HERE
    # Hint:
    # 1. Load pretrained model
    # 2. Freeze ALL parameters
    # 3. Unfreeze parameters in the specified layer
    # 4. Replace and unfreeze classification head
    pass


def get_partial_param_groups(model: nn.Module, config: dict) -> list[dict]:
    """Create parameter groups with different learning rates for Strategy 2.

    Args:
        model: The partially unfrozen model.
        config: Strategy config with 'learning_rates' dict.

    Returns:
        List of param group dicts for the optimizer.
    """
    # YOUR CODE HERE
    # Hint: return [
    #     {'params': model.layer4.parameters(), 'lr': config['learning_rates']['backbone']},
    #     {'params': model.fc.parameters(), 'lr': config['learning_rates']['head']},
    # ]
    pass


# ============================================================
# Strategy 3: Full Fine-tuning with Differential LR
# ============================================================

def create_full_finetune_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    """Create a model for full fine-tuning.

    All parameters are trainable. Uses differential learning rates
    (configured separately in the optimizer).

    Args:
        num_classes: Number of output classes.
        backbone: Pretrained model name.

    Returns:
        Fully trainable pretrained model with new head.
    """
    # YOUR CODE HERE
    # Hint:
    # 1. Load pretrained model
    # 2. Replace classification head
    # 3. All parameters remain trainable (no freezing)
    pass


def get_differential_param_groups(model: nn.Module, config: dict) -> list[dict]:
    """Create parameter groups with differential learning rates for Strategy 3.

    Earlier layers get smaller LRs (features already good).
    Later layers get larger LRs (need more adaptation).

    Args:
        model: The fully trainable model.
        config: Strategy config with 'differential_lr' dict.

    Returns:
        List of param group dicts for the optimizer.
    """
    # YOUR CODE HERE
    # Hint: for ResNet-50:
    # return [
    #     {'params': model.conv1.parameters(),  'lr': config['differential_lr']['conv1']},
    #     {'params': model.bn1.parameters(),    'lr': config['differential_lr']['bn1']},
    #     {'params': model.layer1.parameters(), 'lr': config['differential_lr']['layer1']},
    #     {'params': model.layer2.parameters(), 'lr': config['differential_lr']['layer2']},
    #     {'params': model.layer3.parameters(), 'lr': config['differential_lr']['layer3']},
    #     {'params': model.layer4.parameters(), 'lr': config['differential_lr']['layer4']},
    #     {'params': model.fc.parameters(),     'lr': config['differential_lr']['fc']},
    # ]
    pass


# ============================================================
# Freeze/Unfreeze Helpers
# ============================================================

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head.

    Args:
        model: The model to freeze.
    """
    for name, param in model.named_parameters():
        if "fc" not in name and "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters in the model.

    Args:
        model: The model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: The model to count.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ============================================================
# Factory Function
# ============================================================

def create_model(strategy: str, num_classes: int, config: dict = None) -> nn.Module:
    """Create a model for the given strategy.

    Args:
        strategy: One of 'scratch', 'frozen', 'partial', 'full'.
        num_classes: Number of output classes.
        config: Strategy-specific configuration.

    Returns:
        Configured model.
    """
    config = config or {}
    backbone = config.get("backbone", "resnet50")

    if strategy == "scratch":
        return SimpleCNN(num_classes)
    elif strategy == "frozen":
        return create_frozen_model(num_classes, backbone)
    elif strategy == "partial":
        return create_partial_finetune_model(
            num_classes, backbone,
            unfreeze_from=config.get("unfreeze_from", "layer4"),
        )
    elif strategy == "full":
        return create_full_finetune_model(num_classes, backbone)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
