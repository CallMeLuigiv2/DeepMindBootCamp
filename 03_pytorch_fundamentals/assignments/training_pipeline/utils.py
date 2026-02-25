"""Utility functions for the Training Pipeline project.

100% pre-written -- no stubs. These helpers wrap shared_utils and provide
additional training-specific utilities.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add project root to path for shared_utils access
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import (
    set_seed,
    get_device,
    count_parameters,
    print_model_summary,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
)


# ============================================================
# Training Utilities
# ============================================================

class AverageMeter:
    """Computes and stores the average and current value.

    Usage:
        losses = AverageMeter()
        for batch in loader:
            loss = compute_loss(...)
            losses.update(loss.item(), batch_size)
        print(f"Average loss: {losses.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    """Compute top-k accuracy for the given predictions and targets.

    Args:
        output: Model output logits, shape (batch_size, num_classes).
        target: Ground truth labels, shape (batch_size,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        List of accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item() / batch_size)
        return res


# ============================================================
# Visualization Utilities
# ============================================================

def plot_lr_schedule(
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int,
    steps_per_epoch: int = 1,
    save_path: str = None,
) -> None:
    """Visualize the learning rate schedule over training.

    Args:
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        num_epochs: Number of training epochs.
        steps_per_epoch: Steps per epoch (for per-step schedulers like OneCycleLR).
        save_path: Path to save the figure.
    """
    lrs = []
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            lrs.append(optimizer.param_groups[0]["lr"])
            if hasattr(scheduler, "step"):
                scheduler.step()

    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.xlabel("Step" if steps_per_epoch > 1 else "Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_augmentations(
    dataset,
    n_images: int = 8,
    save_path: str = None,
) -> None:
    """Visualize augmented training images.

    Args:
        dataset: A torchvision dataset with transforms applied.
        n_images: Number of images to display.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    for i in range(n_images):
        img, label = dataset[i]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        # Show same image twice (different augmentations)
        axes[0][i].imshow(img_np)
        axes[0][i].axis("off")
        axes[0][i].set_title(f"Class {label}")

        img2, _ = dataset[i]
        img2_np = img2.numpy().transpose(1, 2, 0)
        img2_np = img2_np * std + mean
        img2_np = np.clip(img2_np, 0, 1)
        axes[1][i].imshow(img2_np)
        axes[1][i].axis("off")

    plt.suptitle("Augmented Training Images (2 views per sample)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Model Analysis
# ============================================================

def compute_model_flops_estimate(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> int:
    """Rough estimate of FLOPs for a forward pass.

    This is a simplified estimate that counts multiply-accumulate operations
    for Conv2d and Linear layers.

    Args:
        model: The model to analyze.
        input_size: Input tensor shape (batch, channels, height, width).

    Returns:
        Estimated FLOPs count.
    """
    total_flops = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * Kh * Kw * Hout * Wout
            out_h, out_w = output.shape[2], output.shape[3]
            flops = 2 * module.in_channels * module.out_channels * \
                    module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
            total_flops += flops
        elif isinstance(module, nn.Linear):
            flops = 2 * module.in_features * module.out_features
            total_flops += flops

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return total_flops
