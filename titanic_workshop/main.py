"""CLI for training and comparing Titanic survival models offline.

This entry point ties together data preparation, model training, and reporting
so workshop participants can run experiments from the command line.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List

from .data import ensure_data, load_dataset, split_features_labels
from .models import (
    ModelResult,
    train_logistic_regression,
    train_neural_network,
    train_xgboost_classifier,
)
from .preprocess import encode_rows, standardize_features, train_test_split

MODEL_CHOICES = ["linear", "xgboost", "nn", "all"]


def run_experiments(models_to_run: List[str], *, data_dir: pathlib.Path, epochs: int) -> List[ModelResult]:
    """Execute the requested models and collect their metrics.

    Args:
        models_to_run: Subset of model identifiers to train.
        data_dir: Directory where the Titanic dataset resides.
        epochs: Number of epochs to use when training the neural network.

    Returns:
        List of ModelResult objects ordered by execution.
    """
    data_path = ensure_data(data_dir)
    df = load_dataset(data_path)
    X_raw, y = split_features_labels(df)
    X_encoded, _ = encode_rows(X_raw)
    X_standardized, means, stds = standardize_features(X_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y)

    results = []
    if "linear" in models_to_run:
        results.append(train_logistic_regression(X_train, y_train, X_test, y_test))
    if "xgboost" in models_to_run:
        results.append(train_xgboost_classifier(X_train, y_train, X_test, y_test))
    if "nn" in models_to_run:
        results.append(
            train_neural_network(
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=epochs,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the workshop CLI."""
    parser = argparse.ArgumentParser(description="Titanic modelling workshop harness")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="all",
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("input"),
        help="Directory where the Titanic CSV is stored or will be downloaded.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs for the neural network (ignored for other models).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training and comparing Titanic models."""
    args = parse_args()
    models = MODEL_CHOICES if args.model == "all" else [args.model]
    results = run_experiments(models, data_dir=args.data_dir, epochs=args.epochs)
    print("\nModel performance\n-----------------")
    for result in results:
        print(
            f"{result.name:16} accuracy={result.accuracy:.3f} roc_auc={result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
