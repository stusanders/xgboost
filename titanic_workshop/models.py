"""Decision-tree-centric models and evaluation helpers."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .preprocess import NumericRow
from .visualize import forest_vote_chart, tree_structure_chart, xgboost_additive_chart


@dataclass
class ModelResult:
    """Container for reporting model evaluation metrics and visuals."""

    name: str
    accuracy: float
    roc_auc: float
    visualizations: list[tuple[str, object]] = field(default_factory=list)


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


@dataclass
class DecisionNode:
    """A minimal binary decision tree node."""

    feature_index: int | None = None
    threshold: float | None = None
    left: "DecisionNode | None" = None
    right: "DecisionNode | None" = None
    prediction: float | None = None
    depth: int = 0

    def predict(self, features: NumericRow) -> float:
        if self.feature_index is None or self.threshold is None or self.left is None or self.right is None:
            return float(self.prediction or 0.0)
        branch = self.right if features[self.feature_index] >= self.threshold else self.left
        return branch.predict(features)


def _gini(groups: list[list[int]]) -> float:
    total = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        if not group:
            continue
        proportion = sum(group) / len(group)
        gini += (1.0 - proportion**2 - (1 - proportion) ** 2) * (len(group) / total)
    return gini


def _best_split(X: List[NumericRow], y: List[int], feature_indices: list[int]) -> tuple[int, float]:
    best_feature = feature_indices[0]
    best_threshold = X[0][best_feature]
    best_score = float("inf")

    for feature in feature_indices:
        thresholds = sorted({row[feature] for row in X})
        for threshold in thresholds:
            left = [label for row, label in zip(X, y) if row[feature] < threshold]
            right = [label for row, label in zip(X, y) if row[feature] >= threshold]
            score = _gini([left, right])
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


def _build_tree(
    X: List[NumericRow],
    y: List[int],
    *,
    max_depth: int,
    min_size: int,
    depth: int,
    feature_selector: Callable[[int], list[int]],
) -> DecisionNode:
    node = DecisionNode(depth=depth)
    if depth >= max_depth or len(X) <= min_size or len(set(y)) == 1:
        node.prediction = sum(y) / len(y) if y else 0.0
        return node

    feature_indices = feature_selector(len(X[0]))
    feature, threshold = _best_split(X, y, feature_indices)
    left_X: list[NumericRow] = []
    left_y: list[int] = []
    right_X: list[NumericRow] = []
    right_y: list[int] = []
    for row, label in zip(X, y):
        if row[feature] < threshold:
            left_X.append(row)
            left_y.append(label)
        else:
            right_X.append(row)
            right_y.append(label)

    node.feature_index = feature
    node.threshold = threshold
    node.left = _build_tree(
        left_X,
        left_y,
        max_depth=max_depth,
        min_size=min_size,
        depth=depth + 1,
        feature_selector=feature_selector,
    )
    node.right = _build_tree(
        right_X,
        right_y,
        max_depth=max_depth,
        min_size=min_size,
        depth=depth + 1,
        feature_selector=feature_selector,
    )
    return node


def train_decision_tree(
    X_train: List[NumericRow],
    y_train: List[int],
    X_test: List[NumericRow],
    y_test: List[int],
    *,
    max_depth: int = 3,
    min_size: int = 2,
    feature_names: list[str] | None = None,
) -> ModelResult:
    """Train a small decision tree using Gini impurity splits."""

    feature_selector = lambda n: list(range(n))
    tree = _build_tree(
        X_train,
        y_train,
        max_depth=max_depth,
        min_size=min_size,
        depth=0,
        feature_selector=feature_selector,
    )

    scores: list[float] = [tree.predict(row) for row in X_test]
    acc, roc = evaluate_predictions(y_test, scores)
    visuals = []
    if feature_names:
        visuals.append(("Decision tree structure", tree_structure_chart(tree, feature_names)))
    return ModelResult("Decision Tree", acc, roc, visuals)


def _bootstrap_sample(X: List[NumericRow], y: List[int]) -> tuple[list[NumericRow], list[int]]:
    indices = [random.randrange(len(X)) for _ in range(len(X))]
    return [X[i] for i in indices], [y[i] for i in indices]


def train_random_forest(
    X_train: List[NumericRow],
    y_train: List[int],
    X_test: List[NumericRow],
    y_test: List[int],
    *,
    n_trees: int = 5,
    max_depth: int = 3,
    min_size: int = 2,
    feature_names: list[str] | None = None,
) -> ModelResult:
    """Train a tiny random forest with feature bagging."""

    trees: list[DecisionNode] = []
    feature_importance = [0.0 for _ in range(len(X_train[0]))]

    for _ in range(n_trees):
        sample_X, sample_y = _bootstrap_sample(X_train, y_train)
        feature_selector = lambda n: random.sample(range(n), max(1, int(math.sqrt(n))))
        tree = _build_tree(
            sample_X,
            sample_y,
            max_depth=max_depth,
            min_size=min_size,
            depth=0,
            feature_selector=feature_selector,
        )
        trees.append(tree)
        feature_importance = [val + 1 for val in feature_importance]

    scores: list[float] = []
    tree_votes: list[dict[str, float]] = []
    for row in X_test:
        preds = [t.predict(row) for t in trees]
        scores.append(sum(preds) / len(preds))
        tree_votes.append({"average_vote": scores[-1]})

    acc, roc = evaluate_predictions(y_test, scores)

    visuals = []
    if feature_names:
        visuals.append(("Forest vote distribution", forest_vote_chart(scores)))
    return ModelResult("Random Forest", acc, roc, visuals)


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
    rounds: int = 10,
    learning_rate: float = 0.3,
    feature_names: list[str] | None = None,
) -> ModelResult:
    """Train a tiny gradient boosting model with decision stumps."""

    pos_ratio = max(1e-6, min(1 - 1e-6, sum(y_train) / len(y_train)))
    base_score = math.log(pos_ratio / (1 - pos_ratio))
    trees: list[Tuple[int, float, float]] = []
    additive_history: list[dict[str, float]] = []

    for round_idx in range(rounds):
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
        additive_history.append({"round": round_idx + 1, "trees": len(trees), "step": weight})

    scores: list[float] = []
    for features in X_test:
        score = base_score
        for feat_idx, threshold, weight in trees:
            score += weight if features[feat_idx] >= threshold else -weight
        scores.append(_sigmoid(score))

    acc, roc = evaluate_predictions(y_test, scores)
    visuals = []
    if feature_names:
        visuals.append(("Boosting contributions", xgboost_additive_chart(additive_history)))
    return ModelResult("XGBoost-lite", acc, roc, visuals)
