"""Training script for the Ensemble Methods Mini-Competition.

Pre-written: argparse, config loading, cross-validation setup, result saving.
Stubbed: training loop body, hyperparameter search logic.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dataset adult --cv-folds 10
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from data import load_dataset, prepare_data, engineer_features
from model import (
    DecisionTreeFromScratch,
    create_random_forest,
    create_gradient_boosting,
    create_voting_ensemble,
    create_stacked_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble Methods Mini-Competition"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name or path (overrides config)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        help="Number of CV folds (overrides config)",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip hyperparameter search",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Run stratified k-fold cross-validation and collect metrics.

    Args:
        model: Sklearn-compatible classifier (must have fit/predict/predict_proba).
        X: Feature matrix.
        y: Labels.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dictionary mapping metric names to lists of per-fold scores.

    Note:
        For DecisionTreeFromScratch, you will need to adapt this to work
        with its fit/predict interface (no predict_proba).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # YOUR CODE HERE: fit model, predict, compute metrics
        # Hint:
        #   model_clone = clone(model)  # from sklearn.base import clone
        #   model_clone.fit(X_train, y_train)
        #   y_pred = model_clone.predict(X_val)
        #   Compute accuracy, precision, recall, f1, roc_auc
        #   Append each to results[metric_name]
        pass

    return results


def run_hyperparameter_search(
    model,
    param_distributions: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 50,
    scoring: str = "f1_weighted",
    cv: int = 5,
    random_state: int = 42,
):
    """Run randomized hyperparameter search.

    Args:
        model: Base estimator.
        param_distributions: Search space (dict of param_name -> list of values).
        X: Feature matrix.
        y: Labels.
        n_iter: Number of random configurations to try.
        scoring: Scoring metric for selection.
        cv: Number of CV folds.
        random_state: Random seed.

    Returns:
        Tuple of (best_estimator, best_params, best_score).
    """
    # YOUR CODE HERE
    # Hint: use RandomizedSearchCV from sklearn
    pass


def format_results(results: dict[str, list[float]]) -> dict[str, str]:
    """Format CV results as 'mean +/- std' strings.

    Args:
        results: Dict from cross_validate_model.

    Returns:
        Dict mapping metric names to formatted strings.
    """
    formatted = {}
    for metric, scores in results.items():
        if len(scores) > 0:
            mean = np.mean(scores)
            std = np.std(scores)
            formatted[metric] = f"{mean:.4f} +/- {std:.4f}"
        else:
            formatted[metric] = "N/A"
    return formatted


def save_results(all_results: dict, path: str) -> None:
    """Save all model results to a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    np.random.seed(args.seed)

    # Override config with CLI args
    dataset_name = args.dataset or config["data"]["dataset"]
    n_folds = args.cv_folds or config["cv"]["n_folds"]

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ENSEMBLE METHODS MINI-COMPETITION")
    print("=" * 60)

    print(f"\nLoading dataset: {dataset_name}")
    df, target_col = load_dataset(dataset_name)
    print(f"  Shape: {df.shape}")
    print(f"  Target: {target_col}")
    print(f"  Target distribution:\n{df[target_col].value_counts()}\n")

    # Feature engineering (implement in data.py)
    print("Engineering features...")
    df = engineer_features(df, target_col)

    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        df, target_col,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}\n")

    # ------------------------------------------------------------------
    # 2. Create models
    # ------------------------------------------------------------------
    models = {}

    # Your decision tree from scratch
    tree_cfg = config["models"]["decision_tree_scratch"]
    models["Decision Tree (yours)"] = DecisionTreeFromScratch(
        max_depth=tree_cfg["max_depth"],
        min_samples_split=tree_cfg["min_samples_split"],
        criterion=tree_cfg.get("criterion", "gini"),
    )

    # sklearn decision tree (sanity check)
    sk_tree_cfg = config["models"]["decision_tree_sklearn"]
    models["Decision Tree (sklearn)"] = DecisionTreeClassifier(
        max_depth=sk_tree_cfg["max_depth"],
        min_samples_split=sk_tree_cfg["min_samples_split"],
        random_state=args.seed,
    )

    # Random Forest
    models["Random Forest"] = create_random_forest(config["models"]["random_forest"])

    # Gradient Boosting
    models["Gradient Boosting"] = create_gradient_boosting(
        config["models"]["gradient_boosting"]
    )

    # Non-tree baseline
    lr_cfg = config["models"]["logistic_regression"]
    models["Logistic Regression"] = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Cross-validate each model
    # ------------------------------------------------------------------
    all_results = {}

    for name, model in models.items():
        print(f"Cross-validating: {name}")
        start = time.time()
        results = cross_validate_model(model, X_train, y_train, n_folds=n_folds)
        elapsed = time.time() - start

        formatted = format_results(results)
        formatted["time_seconds"] = f"{elapsed:.1f}"
        all_results[name] = formatted

        print(f"  Accuracy:  {formatted.get('accuracy', 'N/A')}")
        print(f"  F1:        {formatted.get('f1', 'N/A')}")
        print(f"  ROC-AUC:   {formatted.get('roc_auc', 'N/A')}")
        print(f"  Time:      {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # 4. Hyperparameter search (optional)
    # ------------------------------------------------------------------
    if not args.no_search and config["hyperparameter_search"]["enabled"]:
        print("=" * 60)
        print("HYPERPARAMETER SEARCH")
        print("=" * 60)

        search_cfg = config["hyperparameter_search"]

        # Random Forest search
        print("\nSearching Random Forest hyperparameters...")
        # YOUR CODE HERE: call run_hyperparameter_search for RF
        # Update models["Random Forest"] with best estimator

        # Gradient Boosting search
        print("Searching Gradient Boosting hyperparameters...")
        # YOUR CODE HERE: call run_hyperparameter_search for GB
        # Update models["Gradient Boosting"] with best estimator

        # Re-evaluate best models
        # YOUR CODE HERE

    # ------------------------------------------------------------------
    # 5. Final evaluation on hold-out test set
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # YOUR CODE HERE:
    # For each model:
    #   1. Train on full X_train
    #   2. Predict on X_test
    #   3. Compute and print final metrics
    # Save the best model to checkpoints/

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output_path = config["output"]["results_path"]
    save_results(all_results, output_path)

    print("\nDone! See results in:", output_path)
    print("Next: analyze results in notebooks/analysis.ipynb")


if __name__ == "__main__":
    main()
