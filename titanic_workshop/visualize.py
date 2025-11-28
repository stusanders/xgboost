"""Altair-based visualizations for the tree-centric workshop.

The module attempts to import the Vega-Altair package. If the environment is
offline, a light-weight stub mimics the subset of the API we need so the rest of
the workshop remains runnable.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, List

try:  # pragma: no cover - exercised indirectly
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover - offline fallback
    @dataclass
    class _StubChart:
        data: Any
        spec: dict | None = None

        def mark_circle(self, **kwargs):
            self.spec = {"mark": {"type": "circle", **kwargs}}
            return self

        def mark_line(self, **kwargs):
            self.spec = {"mark": {"type": "line", **kwargs}}
            return self

        def mark_text(self, **kwargs):
            self.spec = {"mark": {"type": "text", **kwargs}}
            return self

        def mark_bar(self, **kwargs):
            self.spec = {"mark": {"type": "bar", **kwargs}}
            return self

        def encode(self, **kwargs):
            self.spec = {**(self.spec or {}), "encoding": kwargs}
            return self

        def properties(self, **kwargs):
            self.spec = {**(self.spec or {}), **kwargs}
            return self

        def interactive(self):
            return self

        def __add__(self, other):
            layer = [self.to_dict()]
            if hasattr(other, "to_dict"):
                layer.append(other.to_dict())
            else:
                layer.append(other)
            return _StubChart(data=None, spec={"layer": layer})

        def to_dict(self):
            return {"data": self.data, **(self.spec or {})}

        def to_json(self):
            return json.dumps(self.to_dict())

    class _StubAlt:
        Chart = _StubChart

        @staticmethod
        def value(val):
            return val

    alt = _StubAlt()  # type: ignore


def tree_structure_chart(tree, feature_names: List[str]):
    """Visualize a decision tree as layered Altair charts."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def walk(node, parent_id: int | None, depth: int, counter: Iterable[int]):
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
        i = 0
        while True:
            yield i
            i += 1

    walk(tree, None, 0, counter())

    node_chart = alt.Chart(nodes).mark_circle(size=150).encode(x="depth:Q", y="y:Q", tooltip=["label"])
    label_chart = alt.Chart(nodes).mark_text(align="left", dx=6).encode(x="depth:Q", y="y:Q", text="label")
    edge_chart = alt.Chart(edges).mark_line(color="#999").encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    return edge_chart + node_chart + label_chart


def forest_vote_chart(votes: List[float]):
    """Show how tree votes spread across the test set."""

    data = [{"index": idx, "vote": vote} for idx, vote in enumerate(votes)]
    return alt.Chart(data).mark_bar().encode(x="index:Q", y="vote:Q", tooltip=["vote"])


def xgboost_additive_chart(history: List[dict[str, float]]):
    """Visualize additive steps of boosting rounds."""

    return (
        alt.Chart(history)
        .mark_line(point=True)
        .encode(x="round:Q", y="step:Q", tooltip=["round", "step", "trees"])
        .properties(title="How each boosting round shifts the model")
    )


def save_visualizations(results: list, output_dir: pathlib.Path) -> None:
    """Persist Altair chart specs as JSON for notebook or Vega usage."""

    output_dir.mkdir(parents=True, exist_ok=True)
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
