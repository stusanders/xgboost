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

Visualization specs are written as Vega/Altair-compatible JSON. After a run
with `--visualize`, open them with the free
[Vega editor](https://vega.github.io/editor/) by importing the JSON files from
`output/visualizations`, or load them into a notebook and render with
`altair.Chart.from_dict(json.load(open("output/visualizations/tree.json")))`.
The files are self-contained, so no network access is required to explore the
charts.

The script downloads the dataset to `input/titanic.csv` if it is not already
present. Use `--data-dir` to change the location. When `--visualize` is
enabled, chart specifications are written to `output/visualizations` by default
so you can load them into notebooks or Vega editors later.

When `--visualize` is enabled, Altair chart JSON files (including a model-wide
metric comparison view) are always written under `output/visualizations` unless
you override the folder with `--visualize-dir`.

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

## Hyperparameter guidance

The CLI exposes a handful of knobs for each model; typical offline-friendly
ranges and their effects are:

- **Decision tree**
  - `--max-depth` (usual range: 2–6): deeper trees memorize quirks in the
    Titanic split and can overfit; shallower trees stay interpretable but may
    miss interactions.
  - `--min-leaf` (usual range: 1–5): larger leaves smooth probabilities and
    protect against noisy splits at the expense of some nuance.
- **Random forest**
  - `--forest-trees` (usual range: 5–50): more trees reduce variance and usually
    improve ROC-AUC until returns diminish; too few trees leave high variance.
  - `--max-depth`/`--min-leaf`: the same controls as the single tree, applied to
    each bootstrapped tree; shallower, pruned trees paired with many trees tend
    to generalize well.
- **XGBoost-lite**
  - `--xgboost-rounds` (usual range: 10–50): additional rounds let the additive
    model correct residuals; excessive rounds with a high learning rate can
    overfit.
  - `--xgboost-lr` (usual range: 0.05–0.5): lower rates take more rounds but can
    reach smoother optima; higher rates move faster but risk oscillations.

Experimenting inside these ranges will influence the evaluation metrics below;
the visualization JSON includes tooltips with the hyperparameters used for each
run so you can see the trade-offs directly.

## Evaluation and visualization

Each model reports accuracy, ROC-AUC, precision, recall, and F1 so you can
balance calibration, ranking quality, and class-specific performance. Lower
precision with high recall often means the model captures more survivors but at
the cost of false positives; high precision with low recall signals conservative
predictions that miss positives. ROC-AUC close to 0.5 indicates near-random
ranking, while scores above ~0.8 reflect meaningful separation on this dataset.

When `--visualize` is used, the workshop saves per-model visualizations plus an
aggregate metric comparison chart to `output/visualizations`, letting you open
the JSON in notebooks or the [Vega editor](https://vega.github.io/editor/) to
see how different hyperparameter choices shift the metrics.

## Repository layout

- `titanic_workshop/data.py`: Dataset download, offline fallback, and feature selection.
- `titanic_workshop/preprocess.py`: Lightweight preprocessing (imputation, encoding) and train/validation split helper.
- `titanic_workshop/models.py`: Decision tree, random forest, and boosted stump trainers plus evaluation helpers.
- `titanic_workshop/visualize.py`: Altair (or fallback) charts explaining trees, forests, and boosting.
- `titanic_workshop/main.py`: CLI to train chosen models, print comparison metrics, and export visualization specs.
- `titanic_workshop/workshop.py`: Thin wrapper delegating to `main.py` for backward compatibility.
- `requirements.txt`: Python dependencies.
