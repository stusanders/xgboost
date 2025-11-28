"""CLI for tree-based Titanic survival models.

This entry point ties together data preparation, model training, visualization,
and reporting so workshop participants can run decision trees, random forests,
and an XGBoost-inspired booster from the command line.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List

from .data import ensure_data, load_dataset, split_features_labels
from .models import ModelResult, train_decision_tree, train_random_forest, train_xgboost_classifier
from .preprocess import encode_rows, train_test_split
from .visualize import metrics_overview_chart, passenger_survival_scatter, save_visualizations

MODEL_CHOICES = ["tree", "forest", "xgboost", "all"]
DEFAULT_VISUALIZE_DIR = pathlib.Path("output/visualizations")


def run_experiments(
    models_to_run: List[str],
    *,
    data_dir: pathlib.Path,
    max_depth: int = 3,
    min_leaf_size: int = 2,
    forest_trees: int = 5,
    xgb_rounds: int = 10,
    xgb_lr: float = 0.3,
    visualize: bool = False,
    visualize_dir: pathlib.Path | None = DEFAULT_VISUALIZE_DIR,
) -> List[ModelResult]:
    """Execute the requested models and collect their metrics.

    Args:
        models_to_run: Subset of model identifiers to train.
        data_dir: Directory where the Titanic dataset resides.
        max_depth: Maximum depth for individual trees.
        min_leaf_size: Minimum sample size before creating a leaf.
        forest_trees: Number of trees to include in the random forest.
        xgb_rounds: Number of boosting rounds for the stump ensemble.
        xgb_lr: Learning rate applied to each stump weight.
        visualize: Whether to persist Altair chart specs alongside metrics.
        visualize_dir: Directory for visualization JSON output. Defaults to
            ``output/visualizations``; if explicitly set to ``None`` the
            location falls back to a ``visualizations`` subfolder alongside the
            data directory.

    Returns:
        List of ModelResult objects ordered by execution.
    """
    data_path = ensure_data(data_dir)
    df = load_dataset(data_path)
    X_raw, y = split_features_labels(df)
    X_encoded, headers = encode_rows(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)

    output_dir = visualize_dir or pathlib.Path(data_dir) / "visualizations"

    results = []
    if "tree" in models_to_run:
        results.append(
            train_decision_tree(
                X_train,
                y_train,
                X_test,
                y_test,
                max_depth=max_depth,
                min_size=min_leaf_size,
                feature_names=headers,
            )
        )
    if "forest" in models_to_run:
        results.append(
            train_random_forest(
                X_train,
                y_train,
                X_test,
                y_test,
                n_trees=forest_trees,
                max_depth=max_depth,
                min_size=min_leaf_size,
                feature_names=headers,
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
                feature_names=headers,
            )
        )

    if visualize:
        summary_visuals = [
            ("Passenger survival", passenger_survival_scatter(df)),
            ("Model metrics", metrics_overview_chart(results)),
        ]
        save_visualizations(results, pathlib.Path(output_dir), summary_visualizations=summary_visuals)
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
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for individual trees.",
    )
    parser.add_argument(
        "--min-leaf",
        type=int,
        default=2,
        help="Minimum records required to form a leaf node.",
    )
    parser.add_argument(
        "--forest-trees",
        type=int,
        default=5,
        help="Number of trees to include in the random forest.",
    )
    parser.add_argument(
        "--xgboost-rounds",
        type=int,
        default=10,
        help="Number of boosting rounds for the stump-based ensemble.",
    )
    parser.add_argument(
        "--xgboost-lr",
        type=float,
        default=0.3,
        help="Learning rate for each boosting step.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save Altair chart specs demonstrating tree behaviour.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=pathlib.Path,
        default=DEFAULT_VISUALIZE_DIR,
        help="Destination directory for visualization JSON (defaults to output/visualizations).",
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
        "Models to run (tree,forest,xgboost,all) [all]: "
    ).strip()
    if not model_input or model_input.lower() == "all":
        models = MODEL_CHOICES[:-1]
    else:
        models = [token.strip().lower() for token in model_input.split(",") if token.strip()]
        models = [m for m in models if m in MODEL_CHOICES]
        if not models:
            models = MODEL_CHOICES[:-1]

    config = {
        "max_depth": _prompt_value("Tree max depth", int, args.max_depth),
        "min_leaf_size": _prompt_value("Leaf minimum size", int, args.min_leaf),
        "forest_trees": _prompt_value("Random forest trees", int, args.forest_trees),
        "xgb_rounds": _prompt_value("XGBoost rounds", int, args.xgboost_rounds),
        "xgb_lr": _prompt_value("XGBoost learning rate", float, args.xgboost_lr),
        "visualize": args.visualize,
        "visualize_dir": args.visualize_dir,
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
            "max_depth": args.max_depth,
            "min_leaf_size": args.min_leaf,
            "forest_trees": args.forest_trees,
            "xgb_rounds": args.xgboost_rounds,
            "xgb_lr": args.xgboost_lr,
            "visualize": args.visualize,
            "visualize_dir": args.visualize_dir,
        }

    results = run_experiments(models, data_dir=args.data_dir, **hyperparams)
    print("\nModel performance\n-----------------")
    for result in results:
        print(
            f"{result.name:16} accuracy={result.accuracy:.3f} "
            f"roc_auc={result.roc_auc:.3f} precision={result.precision:.3f} "
            f"recall={result.recall:.3f} f1={result.f1:.3f}"
        )


if __name__ == "__main__":
    main()
