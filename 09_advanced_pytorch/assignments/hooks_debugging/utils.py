"""Pre-written hook utilities for debugging and instrumentation.

All functions and classes in this file are fully implemented (no stubs).

Provides:
- FeatureExtractor: extract intermediate features via forward hooks
- GradientFlowVisualizer: monitor gradient magnitudes via backward hooks
- ActivationMonitor: track activation statistics via forward hooks
- MagnitudePruner: apply weight pruning via forward pre-hooks
"""

from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor:
    """Extract intermediate features from specified layers using forward hooks.

    Usage:
        extractor = FeatureExtractor(model, ['layer1', 'layer2', 'layer3', 'layer4'])
        features = extractor(input_batch)  # dict mapping layer names to outputs
        extractor.close()  # remove all hooks

    Args:
        model: The model to extract features from.
        layer_names: List of layer names (as returned by model.named_modules()).
    """

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: dict[str, torch.Tensor] = {}
        self._hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self.features[name] = output.detach()
        return hook_fn

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.features.clear()
        with torch.no_grad():
            self.model(x)
        return dict(self.features)

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class GradientFlowVisualizer:
    """Monitor gradient flow through a model using backward hooks.

    Registers backward hooks on every layer with parameters to record
    gradient statistics after each backward pass.

    Usage:
        viz = GradientFlowVisualizer(model)
        loss.backward()
        viz.plot()
        viz.close()

    Args:
        model: The model to monitor.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_stats: dict[str, dict[str, float]] = {}
        self._hooks = []

        for name, module in model.named_modules():
            if self._has_params(module):
                hook = module.register_full_backward_hook(self._make_hook(name))
                self._hooks.append(hook)

    @staticmethod
    def _has_params(module: nn.Module) -> bool:
        return any(p.requires_grad for p in module.parameters(recurse=False))

    def _make_hook(self, name: str):
        def hook_fn(module, grad_input, grad_output):
            grads = []
            for p in module.parameters(recurse=False):
                if p.grad is not None:
                    grads.append(p.grad)
            if grads:
                all_grads = torch.cat([g.flatten() for g in grads])
                self.grad_stats[name] = {
                    "mean": all_grads.abs().mean().item(),
                    "max": all_grads.abs().max().item(),
                    "std": all_grads.std().item(),
                }
        return hook_fn

    def plot(self, title: str = "Gradient Flow", save_path: Optional[str] = None):
        if not self.grad_stats:
            print("No gradient statistics recorded. Run a backward pass first.")
            return

        names = list(self.grad_stats.keys())
        means = [self.grad_stats[n]["mean"] for n in names]
        maxes = [self.grad_stats[n]["max"] for n in names]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
        x = np.arange(len(names))
        ax.bar(x, maxes, alpha=0.3, color="c", label="Max")
        ax.bar(x, means, alpha=0.7, color="b", label="Mean")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
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

    def clear(self):
        self.grad_stats.clear()

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class ActivationMonitor:
    """Monitor activation statistics during training using forward hooks.

    Tracks mean, std, min, max, fraction of zeros, and fraction of negative
    values for all specified layers.

    Usage:
        monitor = ActivationMonitor(model)
        for epoch in range(num_epochs):
            train_one_epoch(...)
            monitor.step()  # record stats snapshot
        monitor.report()
        monitor.plot_over_time()
        monitor.close()

    Args:
        model: The model to monitor.
        layer_types: Tuple of layer types to monitor.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_types: tuple = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Linear, nn.Conv2d),
    ):
        self.model = model
        self._hooks = []
        self._current_stats: dict[str, dict[str, float]] = {}
        self.history: dict[str, list[dict[str, float]]] = defaultdict(list)

        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    flat = output.float().flatten()
                    self._current_stats[name] = {
                        "mean": flat.mean().item(),
                        "std": flat.std().item(),
                        "min": flat.min().item(),
                        "max": flat.max().item(),
                        "frac_zero": (flat == 0).float().mean().item(),
                        "frac_neg": (flat < 0).float().mean().item(),
                    }
        return hook_fn

    def step(self):
        """Record current statistics as a new time step."""
        for name, stats in self._current_stats.items():
            self.history[name].append(dict(stats))
        self._current_stats.clear()

    def report(self):
        """Print a summary table of the latest activation statistics."""
        if not self.history:
            print("No activation statistics recorded.")
            return

        header = f"{'Layer':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%Zero':>8} {'%Neg':>8}"
        print(header)
        print("-" * len(header))

        for name, snapshots in self.history.items():
            latest = snapshots[-1]
            print(
                f"{name:<30} {latest['mean']:>8.4f} {latest['std']:>8.4f} "
                f"{latest['min']:>8.4f} {latest['max']:>8.4f} "
                f"{latest['frac_zero']:>8.2%} {latest['frac_neg']:>8.2%}"
            )

    def plot_over_time(
        self,
        metric: str = "frac_zero",
        title: str = "Activation Statistics Over Time",
        save_path: Optional[str] = None,
    ):
        """Plot a heatmap of activation statistics over training time.

        Args:
            metric: Which metric to plot ('mean', 'std', 'frac_zero', etc.).
            title: Plot title.
            save_path: If provided, save figure to this path.
        """
        if not self.history:
            print("No activation statistics recorded.")
            return

        layer_names = list(self.history.keys())
        num_steps = max(len(v) for v in self.history.values())

        data = np.zeros((len(layer_names), num_steps))
        for i, name in enumerate(layer_names):
            for j, stats in enumerate(self.history[name]):
                data[i, j] = stats[metric]

        fig, ax = plt.subplots(figsize=(max(8, num_steps * 0.3), max(4, len(layer_names) * 0.3)))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=7)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title} ({metric})")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class MagnitudePruner:
    """Magnitude-based weight pruning using forward pre-hooks.

    Registers forward pre-hooks on Linear and Conv2d layers to zero out
    weights below a magnitude threshold before each forward pass.

    Usage:
        pruner = MagnitudePruner(model, prune_ratio=0.5)
        pruner.update_masks()
        # Train or evaluate normally -- masks are applied automatically
        pruner.sparsity_report()
        pruner.close()

    Args:
        model: The model to prune.
        prune_ratio: Fraction of weights to prune (0.0 to 1.0).
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks: dict[str, torch.Tensor] = {}
        self._hooks = []
        self._prunable_layers: dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._prunable_layers[name] = module
                hook = module.register_forward_pre_hook(self._make_hook(name))
                self._hooks.append(hook)

        self.update_masks()

    def _make_hook(self, name: str):
        def hook_fn(module, input):
            if name in self.masks:
                module.weight.data.mul_(self.masks[name])
        return hook_fn

    def update_masks(self):
        """Recompute pruning masks based on current weight magnitudes."""
        for name, module in self._prunable_layers.items():
            weights = module.weight.data.abs().flatten()
            k = int(len(weights) * self.prune_ratio)
            if k > 0:
                threshold = weights.kthvalue(k).values.item()
                self.masks[name] = (module.weight.data.abs() > threshold).float()
            else:
                self.masks[name] = torch.ones_like(module.weight.data)

    def sparsity_report(self):
        """Print the sparsity of each prunable layer."""
        header = f"{'Layer':<40} {'Total':>10} {'Pruned':>10} {'Sparsity':>10}"
        print(header)
        print("-" * len(header))

        total_params = 0
        total_pruned = 0
        for name, mask in self.masks.items():
            n_total = mask.numel()
            n_pruned = (mask == 0).sum().item()
            total_params += n_total
            total_pruned += n_pruned
            print(f"{name:<40} {n_total:>10,} {n_pruned:>10,} {n_pruned/n_total:>10.2%}")

        print("-" * len(header))
        print(f"{'TOTAL':<40} {total_params:>10,} {total_pruned:>10,} {total_pruned/total_params:>10.2%}")

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
