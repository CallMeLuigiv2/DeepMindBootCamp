"""Evaluation metrics — from-scratch implementations and sklearn wrappers."""

from typing import Optional

import numpy as np


# ============================================================
# From-scratch implementations
# ============================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float | np.ndarray:
    """Compute precision.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'binary' for binary classification, 'macro' for unweighted mean across classes.

    Returns:
        Precision score (float for binary/macro, array for per-class).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precisions = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

    if average == "binary":
        return precisions[-1] if len(precisions) > 0 else 0.0
    elif average == "macro":
        return float(np.mean(precisions))
    return np.array(precisions)


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float | np.ndarray:
    """Compute recall.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'binary' for binary classification, 'macro' for unweighted mean across classes.

    Returns:
        Recall score.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    recalls = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    if average == "binary":
        return recalls[-1] if len(recalls) > 0 else 0.0
    elif average == "macro":
        return float(np.mean(recalls))
    return np.array(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float | np.ndarray:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'binary' for binary classification, 'macro' for unweighted mean across classes.

    Returns:
        F1 score.
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    denom = p + r
    if isinstance(denom, np.ndarray):
        f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    else:
        f1 = 2 * p * r / denom if denom > 0 else 0.0

    if average == "macro" and isinstance(f1, np.ndarray):
        return float(np.mean(f1))
    return f1


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC score for binary classification (from scratch).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_scores: Predicted scores/probabilities for the positive class.

    Returns:
        ROC AUC score.
    """
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)

    # Sort by descending score
    desc_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_idx]

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Trapezoidal integration
    return float(np.trapz(tpr, fpr))


# ============================================================
# Sklearn wrappers (convenient when sklearn is available)
# ============================================================

def sklearn_classification_report(y_true, y_pred, **kwargs):
    """Wrapper around sklearn.metrics.classification_report."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, **kwargs)


def sklearn_confusion_matrix(y_true, y_pred, **kwargs) -> np.ndarray:
    """Wrapper around sklearn.metrics.confusion_matrix."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, **kwargs)


def sklearn_roc_auc(y_true, y_scores, **kwargs) -> float:
    """Wrapper around sklearn.metrics.roc_auc_score."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores, **kwargs)
