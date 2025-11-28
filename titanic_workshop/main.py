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


def run_experiments(
    models_to_run: List[str],
    *,
    data_dir: pathlib.Path,
    epochs: int,
    linear_lr: float = 0.1,
    linear_epochs: int = 500,
    xgb_rounds: int = 20,
    xgb_lr: float = 0.3,
    nn_hidden_dim: int = 8,
    nn_lr: float = 0.05,
) -> List[ModelResult]:
    """Execute the requested models and collect their metrics.

    Args:
        models_to_run: Subset of model identifiers to train.
        data_dir: Directory where the Titanic dataset resides.
        epochs: Number of epochs to use when training the neural network.
        linear_lr: Learning rate used by logistic regression.
        linear_epochs: Epoch count for logistic regression gradient descent.
        xgb_rounds: Number of boosting rounds for the stump ensemble.
        xgb_lr: Learning rate applied to each stump weight.
        nn_hidden_dim: Hidden dimension for the neural network.
        nn_lr: Learning rate for neural network weight updates.

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
        results.append(
            train_logistic_regression(
                X_train,
                y_train,
                X_test,
                y_test,
                lr=linear_lr,
                epochs=linear_epochs,
            )
        )
    if "xgboost" in models_to_run:
        results.append(
            train_xgboost_classifier(
                X_train,
                y_train,
                X_test,
                y_test,
                rounds=xgb_rounds,
                learning_rate=xgb_lr,
            )
        )
    if "nn" in models_to_run:
        results.append(
            train_neural_network(
                X_train,
                y_train,
                X_test,
                y_test,
                hidden_dim=nn_hidden_dim,
                epochs=epochs,
                lr=nn_lr,
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
        "--interactive",
        action="store_true",
        help="Prompt for model choices and hyperparameters interactively.",
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
    parser.add_argument(
        "--linear-lr",
        type=float,
        default=0.1,
        help="Learning rate for logistic regression.",
    )
    parser.add_argument(
        "--linear-epochs",
        type=int,
        default=500,
        help="Epochs for logistic regression gradient descent.",
    )
    parser.add_argument(
        "--xgboost-rounds",
        type=int,
        default=20,
        help="Number of boosting rounds for the stump-based ensemble.",
    )
    parser.add_argument(
        "--xgboost-lr",
        type=float,
        default=0.3,
        help="Learning rate for each boosting step.",
    )
    parser.add_argument(
        "--nn-hidden",
        type=int,
        default=8,
        help="Hidden layer width for the neural network.",
    )
    parser.add_argument(
        "--nn-lr",
        type=float,
        default=0.05,
        help="Learning rate for the neural network.",
    )
    return parser.parse_args()


def _prompt_value(prompt: str, caster, default):
    """Request a typed value from stdin, falling back to a default."""

    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return caster(raw)
    except ValueError:
        print("Invalid input, using default.")
        return default


def _interactive_config(args: argparse.Namespace) -> tuple[list[str], dict]:
    """Interactively gather model choices and hyperparameters."""

    model_input = input(
        "Models to run (linear,xgboost,nn,all) [all]: "
    ).strip()
    if not model_input or model_input.lower() == "all":
        models = MODEL_CHOICES[:-1]
    else:
        models = [token.strip().lower() for token in model_input.split(",") if token.strip()]
        models = [m for m in models if m in MODEL_CHOICES]
        if not models:
            models = MODEL_CHOICES[:-1]

    config = {
        "linear_lr": _prompt_value("Logistic regression learning rate", float, args.linear_lr),
        "linear_epochs": _prompt_value("Logistic regression epochs", int, args.linear_epochs),
        "xgb_rounds": _prompt_value("XGBoost rounds", int, args.xgboost_rounds),
        "xgb_lr": _prompt_value("XGBoost learning rate", float, args.xgboost_lr),
        "nn_hidden_dim": _prompt_value("Neural net hidden width", int, args.nn_hidden),
        "epochs": _prompt_value("Neural net epochs", int, args.epochs),
        "nn_lr": _prompt_value("Neural net learning rate", float, args.nn_lr),
    }
    return models, config


def main() -> None:
    """CLI entry point for training and comparing Titanic models."""
    args = parse_args()
    if args.interactive:
        models, hyperparams = _interactive_config(args)
    else:
        models = MODEL_CHOICES[:-1] if args.model == "all" else [args.model]
        hyperparams = {
            "linear_lr": args.linear_lr,
            "linear_epochs": args.linear_epochs,
            "xgb_rounds": args.xgboost_rounds,
            "xgb_lr": args.xgboost_lr,
            "nn_hidden_dim": args.nn_hidden,
            "epochs": args.epochs,
            "nn_lr": args.nn_lr,
        }

    results = run_experiments(models, data_dir=args.data_dir, **hyperparams)
    print("\nModel performance\n-----------------")
    for result in results:
        print(
            f"{result.name:16} accuracy={result.accuracy:.3f} roc_auc={result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
