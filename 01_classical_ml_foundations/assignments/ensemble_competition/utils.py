"""Utility functions for the Ensemble Competition.

100% pre-written -- no stubs. These helpers handle logging, I/O,
and common operations you will use throughout the project.
"""

import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np


# ============================================================
# Timing
# ============================================================

@contextmanager
def timer(description: str = "Operation"):
    """Context manager that prints elapsed time.

    Usage:
        with timer("Training Random Forest"):
            model.fit(X, y)
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{description}] {elapsed:.2f}s")


class Timer:
    """Cumulative timer for tracking time across multiple operations.

    Usage:
        t = Timer()
        t.start("training")
        # ... train model ...
        t.stop("training")
        t.report()
    """

    def __init__(self):
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.time()

    def stop(self, name: str) -> float:
        elapsed = time.time() - self._starts.pop(name)
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        return elapsed

    def report(self) -> None:
        print("\nTiming Report:")
        for name, total in sorted(self._totals.items()):
            print(f"  {name}: {total:.2f}s")


# ============================================================
# Model I/O
# ============================================================

def save_model(model: Any, path: str) -> None:
    """Save a model to disk using pickle.

    Args:
        model: Any picklable model object.
        path: File path (should end in .pkl).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """Load a model from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The loaded model object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Results Logging
# ============================================================

class ResultsLogger:
    """Logger for experiment results. Stores metrics and saves to JSON.

    Usage:
        logger = ResultsLogger()
        logger.log("Random Forest", accuracy=0.95, f1=0.94)
        logger.log("XGBoost", accuracy=0.96, f1=0.95)
        logger.save("results.json")
        logger.print_summary()
    """

    def __init__(self):
        self.results: dict[str, dict[str, Any]] = {}

    def log(self, model_name: str, **metrics) -> None:
        """Log metrics for a model.

        Args:
            model_name: Name of the model.
            **metrics: Keyword arguments of metric_name=value pairs.
        """
        self.results[model_name] = metrics

    def save(self, path: str) -> None:
        """Save results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def load(self, path: str) -> None:
        """Load results from a JSON file."""
        with open(path) as f:
            self.results = json.load(f)

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        if not self.results:
            print("No results logged yet.")
            return

        # Collect all metric names
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        # Header
        header = f"{'Model':<25}" + "".join(f"{m:<18}" for m in all_metrics)
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # Rows
        for model_name, metrics in self.results.items():
            row = f"{model_name:<25}"
            for m in all_metrics:
                val = metrics.get(m, "N/A")
                if isinstance(val, float):
                    row += f"{val:<18.4f}"
                else:
                    row += f"{str(val):<18}"
            print(row)
        print("=" * len(header))


# ============================================================
# Metric Formatting
# ============================================================

def format_cv_scores(scores: list[float], precision: int = 4) -> str:
    """Format cross-validation scores as 'mean +/- std'.

    Args:
        scores: List of per-fold scores.
        precision: Number of decimal places.

    Returns:
        Formatted string like '0.9500 +/- 0.0120'.
    """
    mean = np.mean(scores)
    std = np.std(scores)
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Print and return classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name for the print header.

    Returns:
        Dictionary of metric values.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n--- {model_name} ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics


# ============================================================
# Seed Management
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
