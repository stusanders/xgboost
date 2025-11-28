"""Altair-based visualizations for the tree-centric workshop.

The module attempts to import the Vega-Altair package. If the environment is
offline, a light-weight stub mimics the subset of the API we need so the rest of
the workshop remains runnable.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List

if TYPE_CHECKING:  # pragma: no cover - type hint convenience
    from .models import ModelResult

try:  # pragma: no cover - exercised indirectly
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover - offline fallback
    @dataclass
    class _StubChart:
        """Minimal Altair-compatible object for offline environments."""

        data: Any
        spec: dict | None = None

        def mark_circle(self, **kwargs):
            """Record a circle mark configuration."""

            self.spec = {"mark": {"type": "circle", **kwargs}}
            return self

        def mark_line(self, **kwargs):
            """Record a line mark configuration."""

            self.spec = {"mark": {"type": "line", **kwargs}}
            return self

        def mark_rule(self, **kwargs):
            """Record a rule mark configuration."""

            self.spec = {"mark": {"type": "rule", **kwargs}}
            return self

        def mark_text(self, **kwargs):
            """Record a text mark configuration."""

            self.spec = {"mark": {"type": "text", **kwargs}}
            return self

        def mark_bar(self, **kwargs):
            """Record a bar mark configuration."""

            self.spec = {"mark": {"type": "bar", **kwargs}}
            return self

        def encode(self, **kwargs):
            """Attach encoding details to the chart spec."""

            self.spec = {**(self.spec or {}), "encoding": kwargs}
            return self

        def properties(self, **kwargs):
            """Add presentation properties such as title or dimensions."""

            self.spec = {**(self.spec or {}), **kwargs}
            return self

        def interactive(self):
            """Return the stub to mimic interactive chaining."""

            return self

        def __add__(self, other):
            """Combine two charts into a layered specification."""

            layer = [self.to_dict()]
            if hasattr(other, "to_dict"):
                layer.append(other.to_dict())
            else:
                layer.append(other)
            return _StubChart(data=None, spec={"layer": layer})

        def to_dict(self):
            """Serialize the stub chart to a dictionary."""

            return {"data": self.data, **(self.spec or {})}

        def to_json(self):
            """Serialize the stub chart to a JSON string."""

            return json.dumps(self.to_dict())

    class _StubAlt:
        """Namespace emulating the subset of Altair we rely on."""

        Chart = _StubChart

        @staticmethod
        def value(val):
            """Return literal values for compatibility with Altair syntax."""

            return val

    alt = _StubAlt()  # type: ignore


def tree_structure_chart(tree, feature_names: List[str]):
    """Visualize a decision tree as layered Altair charts.

    Args:
        tree: Root node of the trained decision tree.
        feature_names: Headers describing each feature index in the tree.

    Returns:
        An Altair chart layering edges, nodes, and labels for the tree.
    """

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def walk(node, parent_id: int | None, depth: int, counter: Iterable[int]):
        """Traverse the tree to populate node and edge collections."""

        node_id = next(counter)
        label = (
            f"Leaf: p(1)={node.prediction:.2f}" if node.feature_index is None else f"{feature_names[node.feature_index]} >= {node.threshold:.2f}"
        )
        nodes.append({"id": node_id, "parent": parent_id, "depth": depth, "label": label, "y": len(nodes)})
        if node.left is None or node.right is None:
            return node_id
        left_id = walk(node.left, node_id, depth + 1, counter)
        right_id = walk(node.right, node_id, depth + 1, counter)
        edges.extend(
            [
                {"source": node_id, "target": left_id, "x": depth, "x2": depth + 1, "y": nodes[left_id]["y"], "y2": nodes[left_id]["y"]},
                {"source": node_id, "target": right_id, "x": depth, "x2": depth + 1, "y": nodes[right_id]["y"], "y2": nodes[right_id]["y"]},
            ]
        )
        return node_id

    def counter():
        """Generate incrementing integers for node identifiers."""

        i = 0
        while True:
            yield i
            i += 1

    walk(tree, None, 0, counter())

    node_chart = alt.Chart({"values": nodes}).mark_circle(size=150).encode(
        x="depth:Q", y="y:Q", tooltip=["label:N"], color="depth:Q"
    )
    label_chart = alt.Chart({"values": nodes}).mark_text(align="left", dx=6).encode(
        x="depth:Q", y="y:Q", text="label:N"
    )
    edge_chart = alt.Chart({"values": edges}).mark_line(color="#999").encode(
        x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q"
    )
    return edge_chart + node_chart + label_chart


def forest_vote_chart(votes: List[float]):
    """Show how tree votes spread across the test set.

    Args:
        votes: Probability-style votes contributed by each tree.

    Returns:
        A bar chart visualizing the vote distribution by record index.
    """

    data = [{"index": idx, "vote": vote} for idx, vote in enumerate(votes)]
    return alt.Chart({"values": data}).mark_bar().encode(
        x="index:Q", y="vote:Q", tooltip=["vote:Q"]
    )


def xgboost_additive_chart(history: List[dict[str, float]]):
    """Visualize additive steps of boosting rounds.

    Args:
        history: Sequence of per-round additive contributions produced during
            boosting.

    Returns:
        A line chart depicting how each round shifts the ensemble output.
    """

    return (
        alt.Chart({"values": history})
            .mark_line(point=True)
            .encode(
                x="round:Q",
            y="step:Q",
            tooltip=["round:Q", "step:Q", "trees:Q"],
        )
        .properties(title="How each boosting round shifts the model")
    )


def metrics_overview_chart(results: list["ModelResult"]):
    """Compare core evaluation metrics across trained models.

    Args:
        results: Model evaluations containing metrics and hyperparameters.

    Returns:
        A faceted bar chart summarizing metric values by model.
    """

    records: list[dict[str, Any]] = []
    for result in results:
        hyperparams = ", ".join(f"{k}={v}" for k, v in sorted(result.hyperparameters.items()))
        for metric, value in result.metric_mapping().items():
            records.append(
                {
                    "model": result.name,
                    "metric": metric,
                    "value": value,
                    "hyperparameters": hyperparams or "(defaults)",
                }
            )

    return (
        alt.Chart({"values": records})
            .mark_bar()
            .encode(
                x="metric:N",
            y="value:Q",
            color="model:N",
            column="model:N",
            tooltip=["model:N", "metric:N", "value:Q", "hyperparameters:N"],
        )
        .properties(title="Evaluation metrics across hyperparameter choices")
    )


def passenger_survival_scatter(passengers: list[dict[str, str]]):
    """Show each passenger colored by survival with a 95% split marker.

    Args:
        passengers: Raw Titanic passenger rows including ``Survived`` labels.

    Returns:
        An Altair chart combining passenger points with an optional split rule.
    """

    ordered = sorted(
        enumerate(passengers), key=lambda entry: int(entry[1].get("Survived", "0")), reverse=True
    )

    data: list[dict[str, float | int]] = []
    for idx, (_, row) in enumerate(ordered):
        survived = int(row.get("Survived", 0))
        age_raw = row.get("Age", "")
        try:
            age = float(age_raw) if age_raw not in (None, "") else None
        except ValueError:
            age = None
        data.append(
            {
                "index": idx,
                "survived": survived,
                "age": age,
                "pclass": row.get("Pclass", ""),
                "fare": row.get("Fare", ""),
            }
        )

    survivors = sum(entry["survived"] for entry in data)
    split_index = int(survivors * 0.95) - 1 if survivors else None
    split_chart = None
    if split_index is not None and split_index >= 0:
        split_chart = (
            alt.Chart({"values": [{"position": split_index + 0.5}]})
            .mark_rule(color="#1f77b4", strokeDash=[6, 4])
            .encode(x="position:Q")
        )

    scatter = (
        alt.Chart({"values": data})
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x="index:Q",
            y=alt.Y("age:Q", title="Age (years)"),
            color=alt.Color(
                "survived:N",
                scale={"domain": [1, 0], "range": ["#2ca02c", "#d62728"]},
                legend=alt.Legend(title="Survived"),
            ),
            tooltip=["index:Q", "age:Q", "fare:N", "pclass:N", "survived:N"],
        )
        .properties(title="Passenger survival distribution")
    )

    return scatter if split_chart is None else scatter + split_chart


def save_visualizations(
    results: list["ModelResult"],
    output_dir: pathlib.Path,
    *,
    summary_visualizations: list[tuple[str, object]] | None = None,
) -> None:
    """Persist Altair chart specs as JSON for notebook or Vega usage.

    Args:
        results: Model results containing visualization tuples to export.
        output_dir: Directory where JSON files will be written.
        summary_visualizations: Optional collection of global charts to save.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    for title, chart in summary_visualizations or []:
        safe_name = title.lower().replace(" ", "-")
        target = output_dir / f"summary-{safe_name}.json"
        if hasattr(chart, "to_json"):
            target.write_text(chart.to_json(), encoding="utf-8")
        elif hasattr(chart, "to_dict"):
            import json

            target.write_text(json.dumps(chart.to_dict()), encoding="utf-8")
        else:
            target.write_text(str(chart), encoding="utf-8")

    for result in results:
        for title, chart in result.visualizations:
            safe_name = title.lower().replace(" ", "-")
            target = output_dir / f"{result.name.lower().replace(' ', '-')}-{safe_name}.json"
            if hasattr(chart, "to_json"):
                target.write_text(chart.to_json(), encoding="utf-8")
            elif hasattr(chart, "to_dict"):
                import json

                target.write_text(json.dumps(chart.to_dict()), encoding="utf-8")
            else:
                target.write_text(str(chart), encoding="utf-8")
