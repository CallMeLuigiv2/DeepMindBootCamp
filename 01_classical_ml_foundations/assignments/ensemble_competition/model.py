"""Model definitions for the Ensemble Methods Mini-Competition.

Implement the decision tree from scratch (Part 1) and ensemble wrappers (Part 2).
"""

import numpy as np
from typing import Optional


# ============================================================
# Part 1A: Splitting Criteria
# ============================================================

def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for a set of labels.

    Gini = 1 - sum(p_k^2) for each class k

    Interpretation: probability that two randomly chosen samples
    from the set would have different labels.

    Gini = 0 means perfectly pure (all same class).
    Gini = 0.5 means maximum impurity (for binary classification).

    Args:
        y: Array of class labels, shape (n_samples,).

    Returns:
        Gini impurity score (float between 0 and 1).
    """
    # YOUR CODE HERE
    pass


def entropy(y: np.ndarray) -> float:
    """Compute entropy for a set of labels.

    H = -sum(p_k * log2(p_k)) for each class k

    Interpretation: expected number of bits needed to encode
    a randomly chosen sample's class.

    H = 0 means perfectly pure.
    H = 1 means maximum uncertainty (for binary, 50/50 split).

    Args:
        y: Array of class labels, shape (n_samples,).

    Returns:
        Entropy score (float >= 0).
    """
    # YOUR CODE HERE
    pass


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute information gain from a split.

    IG = H(parent) - weighted_avg(H(left), H(right))

    Higher IG means the split is more useful.

    Args:
        y_parent: Labels before the split.
        y_left: Labels in the left child.
        y_right: Labels in the right child.

    Returns:
        Information gain (float >= 0).
    """
    # YOUR CODE HERE
    pass


# ============================================================
# Part 1B: Finding the Best Split
# ============================================================

def best_split(X: np.ndarray, y: np.ndarray, criterion: str = "gini"):
    """Find the best feature and threshold to split on.

    For each feature:
        For each unique value in that feature (or a subset of thresholds):
            Split the data into left (feature <= threshold) and right (feature > threshold)
            Compute information gain (or Gini reduction)

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        criterion: 'gini' or 'entropy'.

    Returns:
        Tuple of (best_feature_index, best_threshold, best_gain).
        Returns (None, None, 0.0) if no valid split is found.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# Part 1C: Decision Tree from Scratch
# ============================================================

class _TreeNode:
    """Internal tree node representation."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
        value: Optional[int] = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf class prediction

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeFromScratch:
    """Decision tree classifier built from scratch.

    Args:
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        criterion: Splitting criterion ('gini' or 'entropy').
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree: Optional[_TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeFromScratch":
        """Build the tree recursively.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            self
        """
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        """Recursive tree building.

        Base cases: max_depth reached, all samples same class,
                    fewer samples than min_samples_split.
        Recursive case: find best split, partition data, recurse.

        Args:
            X: Feature matrix for the current node.
            y: Labels for the current node.
            depth: Current depth in the tree.

        Returns:
            A _TreeNode (leaf or internal).
        """
        # YOUR CODE HERE
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by traversing the tree for each sample.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted labels, shape (n_samples,).
        """
        # YOUR CODE HERE
        pass

    def _traverse(self, x: np.ndarray, node: _TreeNode) -> int:
        """Traverse the tree for a single sample.

        Args:
            x: Single sample feature vector.
            node: Current tree node.

        Returns:
            Predicted class label.
        """
        # YOUR CODE HERE
        pass


# ============================================================
# Part 2: Ensemble Model Wrappers
# ============================================================

def create_random_forest(config: dict):
    """Create a RandomForestClassifier from config.

    Args:
        config: Dictionary with RF hyperparameters from config.yaml.

    Returns:
        Configured sklearn RandomForestClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 15),
        min_samples_split=config.get("min_samples_split", 5),
        max_features=config.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=-1,
    )


def create_gradient_boosting(config: dict):
    """Create a gradient boosting classifier from config.

    Uses XGBoost if available, falls back to sklearn.

    Args:
        config: Dictionary with GB hyperparameters from config.yaml.

    Returns:
        Configured gradient boosting classifier.
    """
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("learning_rate", 0.1),
            subsample=config.get("subsample", 0.8),
            reg_alpha=config.get("reg_alpha", 0.0),
            reg_lambda=config.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("learning_rate", 0.1),
            subsample=config.get("subsample", 0.8),
            random_state=42,
        )


def create_voting_ensemble(models: dict):
    """Create a VotingClassifier from a dictionary of fitted models.

    Args:
        models: Dict mapping model names to fitted estimator instances.

    Returns:
        Configured sklearn VotingClassifier.

    Note:
        The VotingClassifier must be fit after creation. This function
        only sets up the ensemble -- you still need to call .fit().
    """
    from sklearn.ensemble import VotingClassifier

    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting="soft")


def create_stacked_ensemble(base_models: dict, meta_learner=None):
    """Create a StackingClassifier.

    Args:
        base_models: Dict mapping names to base estimator instances.
        meta_learner: The meta-learner for the final stage.
            Defaults to LogisticRegression if None.

    Returns:
        Configured sklearn StackingClassifier.
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=1000)

    estimators = [(name, model) for name, model in base_models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )
