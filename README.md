# Titanic modelling workshop

This repository contains a coding workshop that compares three models for predicting Titanic passenger survival without external dependencies:

- Logistic regression (linear baseline)
- Gradient-boosted decision stumps (XGBoost-inspired)
- Simple two-layer neural network

The workshop code downloads the public Titanic dataset automatically when network access is available. When offline, it falls back to a bundled sample so participants can still run the exercises end-to-end.

## Setup

Create a Conda environment (or virtualenv) and install the lightweight runtime
dependencies:

```bash
conda env create -f environment.yaml
conda activate titanic-workshop
python -m pip install -r requirements.txt
```

No third-party Python packages are required beyond the standard library, so the
installation steps above simply ensure you have a modern interpreter available.

## Running the workshop

Run all three models and print their metrics:

```bash
python -m titanic_workshop.main --model all
```

For an interactive walkthrough that prompts you to pick models and tweak
hyperparameters from the command line (no code edits required), add
`--interactive`:

```bash
python -m titanic_workshop.main --interactive
```

You can also run models individually:

```bash
python -m titanic_workshop.main --model linear
python -m titanic_workshop.main --model xgboost
python -m titanic_workshop.main --model nn --epochs 40
```

Or pass specific hyperparameters non-interactively:

```bash
python -m titanic_workshop.main \
  --model all \
  --linear-lr 0.05 --linear-epochs 800 \
  --xgboost-rounds 40 --xgboost-lr 0.2 \
  --nn-hidden 16 --nn-lr 0.03 --epochs 120
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

## Hyperparameter tuning cheat sheet

The CLI supports both flags and interactive prompts for key settings. Use these
to experiment with model capacity, learning speed, and regularization:

- **Logistic regression (linear baseline)**
  - `--linear-lr`: Learning rate for gradient updates. Lower values (e.g.,
    `0.01`) train more slowly but can be more stable; higher values (e.g.,
    `0.1`) converge faster but may overshoot.
  - `--linear-epochs`: Number of passes over the data. Increase if the loss
    is still decreasing.
  - `--linear-l1` and `--linear-l2`: Optional regularization strengths. Use a
    small positive value (e.g., `0.001`) to reduce overfitting and encourage
    sparser weights.

- **Gradient-boosted decision stumps (XGBoost-inspired)**
  - `--xgboost-rounds`: Boosting rounds/trees. More rounds increase model
    capacity but can overfit; start with `20–50` and adjust.
  - `--xgboost-lr`: Shrinkage/learning rate applied to each tree’s contribution.
    Lower values (e.g., `0.05`) need more rounds but can generalize better;
    higher values (e.g., `0.3`) learn faster.
  - `--xgboost-subsample`: Fraction of rows to sample for each round. Values
    below `1.0` (e.g., `0.8`) add randomness to reduce overfitting.

- **Two-layer neural network**
  - `--nn-hidden`: Hidden layer width. Larger values increase capacity; try
    `8–64` to balance expressiveness and overfitting risk.
  - `--nn-lr`: Learning rate for stochastic gradient descent. Start around
    `0.01–0.05`; lower if the loss oscillates.
  - `--nn-dropout`: Dropout probability applied to the hidden layer. Use
    `0.0–0.5` to regularize. Higher values reduce overfitting but can slow
    learning.
  - `--epochs`: Training epochs for the neural net (shared with the `--model
    nn` and `--model all` flows). Increase if validation accuracy is still
    improving.

Interactive runs will ask for these values and show sensible defaults. You can
always hit Enter to accept a default, or supply flags directly to script runs
for repeatable experiments.

## Repository layout

- `titanic_workshop/data.py`: Dataset download, offline fallback, and feature selection.
- `titanic_workshop/preprocess.py`: Lightweight preprocessing (imputation, scaling, encoding) and train/validation split helper.
- `titanic_workshop/models.py`: Model definitions, training loops, and evaluation for logistic regression, boosted stumps, and the neural net.
- `titanic_workshop/main.py`: CLI to train chosen models and print comparison metrics.
- `titanic_workshop/workshop.py`: Thin wrapper delegating to `main.py` for backward compatibility.
- `requirements.txt`: Python dependencies.
