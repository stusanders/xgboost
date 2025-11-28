"""Preprocessing helpers for the offline Titanic workshop.

The original implementation relied on pandas and scikit-learn transformers. To
support environments without package installation, this module provides minimal
encoders using only the Python standard library.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

NumericRow = List[float]


def _safe_float(value: str, default: float = 0.0) -> float:
    """Convert a string to float, returning ``default`` on failure."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_rows(rows: Iterable[dict[str, str]]) -> Tuple[list[NumericRow], list[str]]:
    """Encode raw feature dictionaries into numeric vectors.

    Args:
        rows: Iterable of feature dictionaries produced by ``split_features_labels``.

    Returns:
        Tuple of encoded feature vectors and the header order used.
    """

    encoded: list[NumericRow] = []
    # Collect numeric statistics for simple imputation.
    ages: list[float] = []
    fares: list[float] = []
    for row in rows:
        age = _safe_float(row.get("Age", ""))
        fare = _safe_float(row.get("Fare", ""))
        if age:
            ages.append(age)
        if fare:
            fares.append(fare)

    mean_age = sum(ages) / len(ages) if ages else 0.0
    mean_fare = sum(fares) / len(fares) if fares else 0.0

    for row in rows:
        pclass = _safe_float(row.get("Pclass", "0"))
        sex = row.get("Sex", "")
        age = _safe_float(row.get("Age", ""), mean_age)
        sibsp = _safe_float(row.get("SibSp", "0"))
        parch = _safe_float(row.get("Parch", "0"))
        fare = _safe_float(row.get("Fare", ""), mean_fare)
        embarked = row.get("Embarked", "")

        sex_encoded = 1.0 if sex.lower() == "male" else 0.0
        embarked_map: Dict[str, float] = {"S": 0.0, "C": 1.0, "Q": 2.0}
        embarked_encoded = embarked_map.get(embarked, -1.0)

        encoded.append(
            [pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]
        )

    headers = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]
    return encoded, headers


def train_test_split(
    X: Sequence[NumericRow], y: Sequence[int], test_ratio: float = 0.2
) -> Tuple[List[NumericRow], List[NumericRow], List[int], List[int]]:
    """Split the dataset into train and validation partitions.

    This is a deterministic split suitable for workshops and unit testing.
    """

    cutoff = max(1, int(len(X) * (1 - test_ratio)))
    return list(X[:cutoff]), list(X[cutoff:]), list(y[:cutoff]), list(y[cutoff:])


def standardize_features(
    X: List[NumericRow],
) -> Tuple[List[NumericRow], list[float], list[float]]:
    """Standardize each feature to zero mean and unit variance."""

    if not X:
        return [], [], []

    num_features = len(X[0])
    means: list[float] = []
    stds: list[float] = []

    for idx in range(num_features):
        column = [row[idx] for row in X]
        mean = sum(column) / len(column)
        variance = sum((val - mean) ** 2 for val in column) / len(column)
        std = math.sqrt(variance) or 1.0
        means.append(mean)
        stds.append(std)

    standardized: list[NumericRow] = []
    for row in X:
        standardized.append([(row[i] - means[i]) / stds[i] for i in range(num_features)])
    return standardized, means, stds


def apply_standardization(X: List[NumericRow], means: list[float], stds: list[float]) -> List[NumericRow]:
    """Apply precomputed means and standard deviations to a feature matrix."""

    standardized: list[NumericRow] = []
    for row in X:
        standardized.append([(row[i] - means[i]) / stds[i] for i in range(len(row))])
    return standardized
