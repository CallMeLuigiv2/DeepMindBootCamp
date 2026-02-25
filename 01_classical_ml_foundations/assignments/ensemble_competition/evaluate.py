"""Evaluation and visualization for the Ensemble Competition.

Pre-written: model loading, result table formatting, basic plots.
Stubbed: detailed metric computation, comparison visualizations.

Usage:
    python evaluate.py --model-path checkpoints/best_model.pkl
    python evaluate.py --results-path results.json
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from data import load_dataset, prepare_data, engineer_features, analyze_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble models")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a saved model (.pkl)",
    )
    parser.add_argument(
        "--results-path", type=str, default="results.json",
        help="Path to saved results JSON",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Directory to save plots",
    )
    return parser.parse_args()


# ============================================================
# Result Loading and Display (Pre-written)
# ============================================================

def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def print_results_table(results: dict) -> None:
    """Print a formatted comparison table of model results."""
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Accuracy':<18} {'F1':<18} {'ROC-AUC':<18} {'Time (s)':<10}")
    print("=" * 90)
    for model_name, metrics in results.items():
        acc = metrics.get("accuracy", "N/A")
        f1 = metrics.get("f1", "N/A")
        auc = metrics.get("roc_auc", "N/A")
        time_s = metrics.get("time_seconds", "N/A")
        print(f"{model_name:<25} {acc:<18} {f1:<18} {auc:<18} {time_s:<10}")
    print("=" * 90)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to a pandas DataFrame for easier plotting."""
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        for metric, value in metrics.items():
            if isinstance(value, str) and "+/-" in value:
                parts = value.split("+/-")
                row[f"{metric}_mean"] = float(parts[0].strip())
                row[f"{metric}_std"] = float(parts[1].strip())
            else:
                row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Visualization (Stubbed)
# ============================================================

def plot_model_comparison(results: dict, metric: str = "f1", save_path: str = None) -> None:
    """Create a bar chart comparing models on a given metric.

    Args:
        results: Results dictionary from training.
        metric: Metric name to compare (e.g., 'accuracy', 'f1', 'roc_auc').
        save_path: Path to save the figure. If None, display only.
    """
    # YOUR CODE HERE
    # Hint:
    #   - Extract mean and std for each model
    #   - Use plt.bar with yerr for error bars
    #   - Label axes, add title
    pass


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20,
                            save_path: str = None) -> None:
    """Plot feature importances as a horizontal bar chart.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to show.
        save_path: Path to save the figure.
    """
    # YOUR CODE HERE
    pass


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list[str] = None,
                          save_path: str = None) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class label names.
        save_path: Path to save the figure.
    """
    # YOUR CODE HERE
    pass


def plot_depth_analysis(tree_model_class, X_train, y_train, X_test, y_test,
                        max_depths: list[int] = None,
                        save_path: str = None) -> None:
    """Plot train vs test accuracy as a function of tree depth.

    This demonstrates the bias-variance tradeoff.

    Args:
        tree_model_class: Decision tree class (yours or sklearn).
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        max_depths: List of max_depth values to try.
        save_path: Path to save the figure.
    """
    # YOUR CODE HERE
    # Hint:
    #   For each depth in max_depths:
    #     Train a tree, compute train and test accuracy
    #   Plot both curves on the same axes
    pass


def plot_learning_rate_interaction(save_path: str = None) -> None:
    """Plot the interaction between learning_rate and n_estimators for gradient boosting.

    For bias-variance analysis in Part 3.

    Args:
        save_path: Path to save the figure.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and display results
    if os.path.exists(args.results_path):
        results = load_results(args.results_path)
        print_results_table(results)

        # Generate comparison plots
        plot_model_comparison(
            results, metric="f1",
            save_path=os.path.join(args.output_dir, "model_comparison_f1.png"),
        )
        plot_model_comparison(
            results, metric="accuracy",
            save_path=os.path.join(args.output_dir, "model_comparison_accuracy.png"),
        )

    # Load a specific model for detailed evaluation
    if args.model_path and os.path.exists(args.model_path):
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)
        print(f"\nLoaded model from {args.model_path}")

        # YOUR CODE HERE:
        # 1. Load test data
        # 2. Compute predictions
        # 3. Print classification report
        # 4. Plot confusion matrix
        # 5. Plot feature importances

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
