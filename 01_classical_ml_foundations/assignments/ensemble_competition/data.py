"""Data loading and feature engineering for the Ensemble Competition.

Pre-written: dataset downloading, basic loading, train/test splitting.
Stubbed: feature engineering functions for you to implement.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# Dataset Loaders (Pre-written)
# ============================================================

def load_adult_income(data_dir: str = "data") -> tuple[pd.DataFrame, str]:
    """Load the UCI Adult Income dataset.

    Downloads from the UCI ML repository if not cached locally.

    Args:
        data_dir: Directory to cache downloaded data.

    Returns:
        Tuple of (DataFrame, target_column_name).
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "adult.csv")

    if not os.path.exists(filepath):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income",
        ]
        df = pd.read_csv(url, names=columns, sep=r",\s*", engine="python", na_values="?")
        df.to_csv(filepath, index=False)
        print(f"Downloaded Adult Income dataset to {filepath}")
    else:
        df = pd.read_csv(filepath)

    return df, "income"


def load_dataset(name: str, data_dir: str = "data") -> tuple[pd.DataFrame, str]:
    """Load a dataset by name.

    Args:
        name: Dataset identifier. Supported: 'adult'.
            For custom datasets, pass a path to a CSV file.
        data_dir: Directory to cache downloaded data.

    Returns:
        Tuple of (DataFrame, target_column_name).
    """
    if name == "adult":
        return load_adult_income(data_dir)
    elif os.path.isfile(name):
        df = pd.read_csv(name)
        # Assume the last column is the target
        target = df.columns[-1]
        return df, target
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            "Use 'adult' or provide a path to a CSV file."
        )


# ============================================================
# Preprocessing (Pre-written)
# ============================================================

def encode_categoricals(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features using LabelEncoder.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (encoded DataFrame, dict mapping column names to encoders).
    """
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Handle missing values in the dataset.

    Args:
        df: Input DataFrame.
        strategy: 'median' for numerical, mode for categorical.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64, float, int]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Full preprocessing pipeline: handle missing values, encode, split.

    Args:
        df: Raw DataFrame.
        target_col: Name of the target column.
        test_size: Fraction for the hold-out test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    """
    df = handle_missing_values(df)
    df, _ = encode_categoricals(df, target_col)

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(np.float64)
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    return X_train, X_test, y_train, y_test, feature_names


# ============================================================
# Feature Engineering (Stubbed -- YOUR CODE HERE)
# ============================================================

def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Engineer new features from the raw dataset.

    Requirements (at least 5 engineered features):
    - Create interaction features (e.g., feature_A * feature_B)
    - Create aggregate features (e.g., ratios, sums of related features)
    - Apply transformations (log, square root, binning) where appropriate
    - Document each feature: what it is, why you think it might help

    Args:
        df: Raw DataFrame (before encoding).
        target_col: Name of the target column.

    Returns:
        DataFrame with original + engineered features.
    """
    df = df.copy()

    # Example (for Adult Income dataset -- adapt for your chosen dataset):
    #
    # Feature 1: age * hours_per_week (interaction)
    #   Hypothesis: older workers who work many hours may earn more
    # YOUR CODE HERE

    # Feature 2: capital_gain - capital_loss (net capital)
    #   Hypothesis: net investment returns are more predictive than raw values
    # YOUR CODE HERE

    # Feature 3: log(capital_gain + 1)
    #   Hypothesis: capital gain is heavily skewed; log transform may help
    # YOUR CODE HERE

    # Feature 4: hours_per_week binned (part-time / full-time / overtime)
    #   Hypothesis: categorical grouping may capture non-linear effects
    # YOUR CODE HERE

    # Feature 5: education_num / age (education rate)
    #   Hypothesis: higher education relative to age may signal career trajectory
    # YOUR CODE HERE

    return df


def analyze_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract and rank feature importances from a trained model.

    Args:
        model: A fitted model with a `feature_importances_` attribute.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)
