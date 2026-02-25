"""Plotting utilities for visualizing training, evaluation, and model internals."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_training_curves(
    train_losses: list[float],
    val_losses: Optional[list[float]] = None,
    train_accs: Optional[list[float]] = None,
    val_accs: Optional[list[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch (optional).
        train_accs: Training accuracy per epoch (optional).
        val_accs: Validation accuracy per epoch (optional).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    has_acc = train_accs is not None
    fig, axes = plt.subplots(1, 1 + int(has_acc), figsize=(6 * (1 + int(has_acc)), 4))
    if not has_acc:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        ax = axes[1]
        ax.plot(epochs, train_accs, label="Train Acc")
        if val_accs is not None:
            ax.plot(epochs, val_accs, label="Val Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        class_names: List of class label names.
        title: Plot title.
        cmap: Matplotlib colormap name.
        save_path: If provided, save figure to this path.
    """
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_weights(
    attention: np.ndarray,
    x_labels: Optional[list[str]] = None,
    y_labels: Optional[list[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
) -> None:
    """Plot attention weight heatmap.

    Args:
        attention: Attention weights of shape (query_len, key_len) or (n_heads, query_len, key_len).
        x_labels: Labels for key positions.
        y_labels: Labels for query positions.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    if attention.ndim == 3:
        n_heads = attention.shape[0]
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).flatten()
        for i in range(n_heads):
            im = axes[i].imshow(attention[i], cmap="viridis", aspect="auto")
            axes[i].set_title(f"Head {i}")
            if x_labels:
                axes[i].set_xticks(range(len(x_labels)))
                axes[i].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
            if y_labels:
                axes[i].set_yticks(range(len(y_labels)))
                axes[i].set_yticklabels(y_labels, fontsize=7)
        for i in range(n_heads, len(axes)):
            axes[i].axis("off")
        fig.suptitle(title, fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attention, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gradients(
    model,
    title: str = "Gradient Flow",
    save_path: Optional[str] = None,
) -> None:
    """Plot gradient magnitudes per layer (call after loss.backward()).

    Args:
        model: PyTorch model (after backward pass).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    layers = []
    avg_grads = []
    max_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.5), 5))
    x = np.arange(len(layers))
    ax.bar(x, max_grads, alpha=0.3, lw=1, color="c", label="Max")
    ax.bar(x, avg_grads, alpha=0.7, lw=1, color="b", label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_images_grid(
    images: np.ndarray,
    n_rows: int = 2,
    n_cols: int = 8,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot a grid of images (useful for generated samples, reconstructions, etc.).

    Args:
        images: Array of shape (N, H, W) or (N, H, W, C).
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
