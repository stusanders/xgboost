# Titanic modelling workshop

This repository contains a coding workshop that compares three models for predicting Titanic passenger survival without external dependencies:

- Logistic regression (linear baseline)
- Gradient-boosted decision stumps (XGBoost-inspired)
- Simple two-layer neural network

The workshop code downloads the public Titanic dataset automatically when network access is available. When offline, it falls back to a bundled sample so participants can still run the exercises end-to-end.

## Setup

No external packages are required beyond the Python standard library.

## Running the workshop

Run all three models and print their metrics:

```bash
python -m titanic_workshop.main --model all
```

You can also run models individually:

```bash
python -m titanic_workshop.main --model linear
python -m titanic_workshop.main --model xgboost
python -m titanic_workshop.main --model nn --epochs 40
```

The script downloads the dataset to `input/titanic.csv` if it is not already present. Use `--data-dir` to change the location.

## Testing

Run a quick smoke test that executes all models end-to-end using a temporary
data directory:

```bash
python -m unittest
```

## What to explore during the workshop

- Try feature engineering (e.g., family size, title extraction) by editing `titanic_workshop/data.py` and `titanic_workshop/preprocess.py`.
- Tune hyperparameters: regularization strength for logistic regression, depth/learning rate/trees for XGBoost, and layers/dropout/learning rate for the neural network.
- Compare speed and overfitting: XGBoost and the neural net can capture non-linear relationships; logistic regression offers interpretability.

## Repository layout

- `titanic_workshop/data.py`: Dataset download, offline fallback, and feature selection.
- `titanic_workshop/preprocess.py`: Lightweight preprocessing (imputation, scaling, encoding) and train/validation split helper.
- `titanic_workshop/models.py`: Model definitions, training loops, and evaluation for logistic regression, boosted stumps, and the neural net.
- `titanic_workshop/main.py`: CLI to train chosen models and print comparison metrics.
- `titanic_workshop/workshop.py`: Thin wrapper delegating to `main.py` for backward compatibility.
- `requirements.txt`: Python dependencies.
