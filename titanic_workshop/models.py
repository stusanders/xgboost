"""Model definitions and training routines without external dependencies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .preprocess import NumericRow


@dataclass
class ModelResult:
    """Container for reporting model evaluation metrics."""

    name: str
    accuracy: float
    roc_auc: float


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _roc_auc(labels: List[int], scores: List[float]) -> float:
    """Compute ROC-AUC using a ranking algorithm."""

    paired = sorted(zip(scores, labels), key=lambda pair: pair[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5

    rank_sum = 0.0
    for rank, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += rank
    u = rank_sum - pos * (pos + 1) / 2
    return u / (pos * neg)


def evaluate_predictions(y_true: List[int], y_proba: List[float]) -> Tuple[float, float]:
    preds = [1 if p >= 0.5 else 0 for p in y_proba]
    correct = sum(int(p == t) for p, t in zip(preds, y_true))
    accuracy = correct / len(y_true) if y_true else 0.0
    roc_auc = _roc_auc(y_true, y_proba)
    return accuracy, roc_auc


def train_logistic_regression(
    X_train: List[NumericRow],
    y_train: List[int],
    X_test: List[NumericRow],
    y_test: List[int],
    *,
    lr: float = 0.1,
    epochs: int = 500,
) -> ModelResult:
    """Train a simple logistic regression via gradient descent."""

    weights = [0.0 for _ in range(len(X_train[0]))]
    bias = 0.0

    for _ in range(epochs):
        for features, label in zip(X_train, y_train):
            linear = sum(w * x for w, x in zip(weights, features)) + bias
            pred = _sigmoid(linear)
            error = pred - label
            for i in range(len(weights)):
                weights[i] -= lr * error * features[i]
            bias -= lr * error

    scores: list[float] = []
    for features in X_test:
        linear = sum(w * x for w, x in zip(weights, features)) + bias
        scores.append(_sigmoid(linear))

    acc, roc = evaluate_predictions(y_test, scores)
    return ModelResult("Logistic Regression", acc, roc)


def _best_stump(X: List[NumericRow], residuals: List[float]) -> Tuple[int, float]:
    """Find the best single-feature threshold to reduce squared error."""

    best_feature = 0
    best_threshold = 0.0
    best_error = float("inf")

    num_features = len(X[0])
    for feature in range(num_features):
        thresholds = sorted({row[feature] for row in X})
        for threshold in thresholds:
            preds = [1.0 if row[feature] >= threshold else -1.0 for row in X]
            error = sum((r - p) ** 2 for r, p in zip(residuals, preds))
            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


def train_xgboost_classifier(
    X_train: List[NumericRow],
    y_train: List[int],
    X_test: List[NumericRow],
    y_test: List[int],
    *,
    rounds: int = 20,
    learning_rate: float = 0.3,
) -> ModelResult:
    """Train a tiny gradient boosting model with decision stumps."""

    # Initialize with log-odds of the positive class.
    pos_ratio = max(1e-6, min(1 - 1e-6, sum(y_train) / len(y_train)))
    base_score = math.log(pos_ratio / (1 - pos_ratio))
    trees: list[Tuple[int, float, float]] = []

    # Use gradient boosting on logistic loss.
    for _ in range(rounds):
        preds = []
        for features in X_train:
            score = base_score
            for feat_idx, threshold, weight in trees:
                score += weight if features[feat_idx] >= threshold else -weight
            preds.append(_sigmoid(score))

        residuals = [label - pred for label, pred in zip(y_train, preds)]
        best_feature, best_threshold = _best_stump(X_train, residuals)
        weight = learning_rate
        trees.append((best_feature, best_threshold, weight))

    # Evaluate
    scores: list[float] = []
    for features in X_test:
        score = base_score
        for feat_idx, threshold, weight in trees:
            score += weight if features[feat_idx] >= threshold else -weight
        scores.append(_sigmoid(score))

    acc, roc = evaluate_predictions(y_test, scores)
    return ModelResult("XGBoost-lite", acc, roc)


def train_neural_network(
    X_train: List[NumericRow],
    y_train: List[int],
    X_test: List[NumericRow],
    y_test: List[int],
    *,
    hidden_dim: int = 8,
    epochs: int = 200,
    lr: float = 0.05,
) -> ModelResult:
    """Train a two-layer neural network using manual gradient descent."""

    input_dim = len(X_train[0])
    rng = 0.1
    w1 = [[(i + j + 1) * rng / (input_dim + hidden_dim) for j in range(hidden_dim)] for i in range(input_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [rng for _ in range(hidden_dim)]
    b2 = 0.0

    def relu(x: float) -> float:
        return x if x > 0 else 0.0

    def relu_deriv(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    for _ in range(epochs):
        for features, label in zip(X_train, y_train):
            # Forward pass
            hidden_raw = [
                sum(features[i] * w1[i][j] for i in range(input_dim)) + b1[j]
                for j in range(hidden_dim)
            ]
            hidden = [relu(x) for x in hidden_raw]
            output_raw = sum(hidden[j] * w2[j] for j in range(hidden_dim)) + b2
            pred = _sigmoid(output_raw)

            # Backpropagation
            error = pred - label
            d_output = error * pred * (1 - pred)

            d_w2 = [d_output * h for h in hidden]
            d_b2 = d_output

            d_hidden = [d_output * w2[j] * relu_deriv(hidden_raw[j]) for j in range(hidden_dim)]
            d_w1 = [
                [d_hidden[j] * features[i] for j in range(hidden_dim)]
                for i in range(input_dim)
            ]
            d_b1 = d_hidden

            # Update weights
            for j in range(hidden_dim):
                w2[j] -= lr * d_w2[j]
            b2 -= lr * d_b2

            for i in range(input_dim):
                for j in range(hidden_dim):
                    w1[i][j] -= lr * d_w1[i][j]
            for j in range(hidden_dim):
                b1[j] -= lr * d_b1[j]

    # Evaluation
    scores: list[float] = []
    for features in X_test:
        hidden_raw = [
            sum(features[i] * w1[i][j] for i in range(input_dim)) + b1[j]
            for j in range(hidden_dim)
        ]
        hidden = [relu(x) for x in hidden_raw]
        output_raw = sum(hidden[j] * w2[j] for j in range(hidden_dim)) + b2
        scores.append(_sigmoid(output_raw))

    acc, roc = evaluate_predictions(y_test, scores)
    return ModelResult("Neural Network", acc, roc)
