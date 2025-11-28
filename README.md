# Titanic decision tree workshop

This repository focuses on a tree-ensemble journey for predicting Titanic
survival while staying offline-friendly:

- Decision trees: build an interpretable path of splits and visualize it.
- Random forests: explore bagging and feature randomness to stabilize trees.
- XGBoost-inspired boosting: finish with an additive stump ensemble and compare
  it to bagging.

The workshop downloads the public Titanic dataset automatically when network
access is available. When offline, it falls back to a bundled sample so
participants can still run the exercises end-to-end.

## Setup

Create a Conda environment (or virtualenv) and install the lightweight runtime
dependencies:

```bash
conda env create -f environment.yaml
conda activate titanic-workshop
python -m pip install -r requirements.txt
```

No third-party Python packages are required for core training. Optional
visualizations use [Vega-Altair](https://altair-viz.github.io/); a tiny stub is
bundled for offline execution.

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
python -m titanic_workshop.main --model tree
python -m titanic_workshop.main --model forest --forest-trees 8
python -m titanic_workshop.main --model xgboost
```

Or pass specific hyperparameters non-interactively and write visualizations to
the default ``output/visualizations`` folder:

```bash
python -m titanic_workshop.main \
  --model all \
  --max-depth 4 --min-leaf 2 \
  --forest-trees 15 \
  --xgboost-rounds 25 --xgboost-lr 0.2 \
  --visualize
```

The script downloads the dataset to `input/titanic.csv` if it is not already
present. Use `--data-dir` to change the location. When `--visualize` is
enabled, chart specifications are written to `output/visualizations` by default
so you can load them into notebooks or Vega editors later.

## Testing

Run a quick smoke test that executes all models end-to-end using a temporary
data directory:

```bash
python -m unittest
```

## Workshop walkthrough

- Start with a **decision tree**: inspect splits, read leaf probabilities, and
  export the structure as an Altair JSON spec via `--visualize`.
- Move to a **random forest**: adjust `--forest-trees` and see how combining
  bootstrapped trees stabilizes validation metrics. The forest vote chart shows
  how individual trees average out on the hold-out set.
- Finish with **XGBoost-like boosting**: tweak `--xgboost-rounds` and
  `--xgboost-lr` to see how additive steps differ from bagging. The boosting
  visualization illustrates how each round nudges the model.

Interactive runs will ask for these values and show sensible defaults. You can
always hit Enter to accept a default, or supply flags directly to script runs
for repeatable experiments.

## Repository layout

- `titanic_workshop/data.py`: Dataset download, offline fallback, and feature selection.
- `titanic_workshop/preprocess.py`: Lightweight preprocessing (imputation, encoding) and train/validation split helper.
- `titanic_workshop/models.py`: Decision tree, random forest, and boosted stump trainers plus evaluation helpers.
- `titanic_workshop/visualize.py`: Altair (or fallback) charts explaining trees, forests, and boosting.
- `titanic_workshop/main.py`: CLI to train chosen models, print comparison metrics, and export visualization specs.
- `titanic_workshop/workshop.py`: Thin wrapper delegating to `main.py` for backward compatibility.
- `requirements.txt`: Python dependencies.
