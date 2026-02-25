"""Evaluation and Grad-CAM visualization for the Transfer Learning project.

Pre-written: Grad-CAM implementation, visualization helpers, confusion matrix.
Stubbed: strategy comparison visualization.

Usage:
    python evaluate.py --checkpoint checkpoints/full_best.pth --strategy full --gradcam
    python evaluate.py --checkpoint checkpoints/frozen_best.pth --strategy frozen
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.common import get_device, load_checkpoint

from model import create_model
from data import load_dataset, unnormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transfer Learning Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="full",
                        choices=["scratch", "frozen", "partial", "full"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualizations")
    parser.add_argument("--n-images", type=int, default=10, help="Number of Grad-CAM images")
    parser.add_argument("--output-dir", type=str, default="figures")
    return parser.parse_args()


# ============================================================
# Grad-CAM Implementation (Pre-written)
# ============================================================

class GradCAM:
    """Grad-CAM implementation for model interpretability.

    Generates class-discriminative localization maps by using the
    gradient of the target class flowing into the final convolutional layer.

    Usage:
        cam = GradCAM(model, model.layer4[-1])
        heatmap, pred_class = cam.generate(input_tensor)
        cam.remove_hooks()

    Args:
        model: Trained model.
        target_layer: The convolutional layer to visualize
            (e.g., model.layer4[-1] for ResNet).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Class index to generate heatmap for.
                If None, uses the predicted class.

        Returns:
            Tuple of (heatmap, predicted_class).
            Heatmap is a numpy array of shape (H, W) in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        )

        return cam.squeeze().cpu().numpy(), target_class

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# Visualization Helpers (Pre-written)
# ============================================================

def visualize_gradcam(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    predicted_class: int,
    true_class: int,
    class_names: list[str],
    save_path: str = None,
) -> None:
    """Visualize Grad-CAM results: original, heatmap, and overlay.

    Args:
        image_tensor: Normalized image tensor of shape (1, C, H, W) or (C, H, W).
        cam: Grad-CAM heatmap array of shape (H, W) in [0, 1].
        predicted_class: Predicted class index.
        true_class: True class index.
        class_names: List of class names.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img = unnormalize(image_tensor.squeeze())

    # Original
    axes[0].imshow(img)
    true_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    axes[0].set_title(f"True: {true_name}")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(cam, cmap="jet")
    pred_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
    axes[1].set_title(f"Pred: {pred_name}")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    correct = "CORRECT" if predicted_class == true_class else "WRONG"
    axes[2].set_title(f"Overlay ({correct})")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str = None,
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Class names.
        save_path: Path to save figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True", xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Evaluation (Stubbed for per-class analysis)
# ============================================================

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Full evaluation with per-class metrics.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        device: Device.
        class_names: Class names.

    Returns:
        Tuple of (overall_accuracy, all_preds, all_labels).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return accuracy, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    data_config = config.get("data", {})
    data_config.setdefault("batch_size", 32)
    _, _, test_loader, class_names, num_classes = load_dataset(data_config)

    # Load model
    print(f"Loading {args.strategy} model from {args.checkpoint}")
    model = create_model(args.strategy, num_classes, config.get("model", {})).to(device)
    ckpt = load_checkpoint(args.checkpoint, model)
    print(f"  Best val acc: {ckpt.get('best_val_acc', '?')}")

    # Evaluate
    accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device, class_names)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Per-class report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names[:num_classes] if len(class_names) >= num_classes else None,
        zero_division=0,
    ))

    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names[:num_classes],
        save_path=os.path.join(args.output_dir, f"confusion_{args.strategy}.png"),
    )

    # Grad-CAM
    if args.gradcam:
        print(f"\nGenerating Grad-CAM for {args.n_images} images...")

        # Get target layer for Grad-CAM
        if args.strategy == "scratch":
            # For SimpleCNN, use the last conv layer
            target_layer = model.features[-4]  # Last Conv2d before GAP
        else:
            # For ResNet-50, use the last residual block
            target_layer = model.layer4[-1]

        gradcam = GradCAM(model, target_layer)

        # Get test images
        test_iter = iter(test_loader)
        images, labels = next(test_iter)
        n = min(args.n_images, len(images))

        for i in range(n):
            img = images[i].unsqueeze(0).to(device)
            true_label = labels[i].item()

            cam, pred_class = gradcam.generate(img)

            save_path = os.path.join(
                args.output_dir,
                f"gradcam_{args.strategy}_{i}_true{true_label}_pred{pred_class}.png",
            )
            visualize_gradcam(
                images[i], cam, pred_class, true_label, class_names,
                save_path=save_path,
            )

        gradcam.remove_hooks()
        print(f"Grad-CAM images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
