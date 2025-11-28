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

            result = {**(self.spec or {})}
            if self.data is not None:
                result["data"] = self.data
            return result

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


def _humanize_feature_name(feature_name: str) -> str:
    """Convert technical feature names to human-readable names.

    Args:
        feature_name: Technical feature name from the dataset.

    Returns:
        Human-readable feature name.
    """
    name_map = {
        "Pclass": "Passenger Class",
        "Sex": "Sex",
        "Age": "Age",
        "SibSp": "Siblings/Spouses",
        "Parch": "Parents/Children",
        "Fare": "Fare",
        "Embarked": "Embarked",
    }
    return name_map.get(feature_name, feature_name)


def _humanize_split(feature_name: str, threshold: float, direction: str) -> tuple[str, str]:
    """Convert a split condition into human-readable labels.

    Args:
        feature_name: Name of the feature being split on.
        threshold: Threshold value for the split.
        direction: Direction of the split ("left" for <, "right" for >=).

    Returns:
        Tuple of (short_label, detailed_label) for display.
    """
    # Handle Sex feature (binary encoded: 1.0 = male, 0.0 = female)
    if feature_name == "Sex":
        if direction == "right":  # >= threshold
            if threshold <= 0.5:
                return "Female", "Female passengers"
            else:
                return "Male", "Male passengers"
        else:  # < threshold
            if threshold <= 0.5:
                return "Male", "Male passengers"
            else:
                return "Female", "Female passengers"

    # Handle Pclass (1.0 = First, 2.0 = Second, 3.0 = Third)
    elif feature_name == "Pclass":
        if direction == "right":  # >= threshold
            if threshold <= 1.5:
                return "1st Class", "1st Class passengers"
            elif threshold <= 2.5:
                return "2nd/3rd Class", "2nd and 3rd Class passengers"
            else:
                return "3rd Class", "3rd Class passengers"
        else:  # < threshold
            if threshold <= 1.5:
                return "< 1st Class", "Below 1st Class"
            elif threshold <= 2.5:
                return "1st Class", "1st Class passengers"
            else:
                return "1st/2nd Class", "1st and 2nd Class passengers"

    # Handle Age (continuous)
    elif feature_name == "Age":
        age_int = int(threshold)
        if direction == "right":
            return f"Age {age_int}+", f"Age {age_int} or older"
        else:
            return f"Age < {age_int}", f"Age under {age_int}"

    # Handle Fare (continuous, in pounds)
    elif feature_name == "Fare":
        if direction == "right":
            return f"Fare £{threshold:.1f}+", f"Fare £{threshold:.2f} or more"
        else:
            return f"Fare < £{threshold:.1f}", f"Fare less than £{threshold:.2f}"

    # Handle SibSp (number of siblings/spouses)
    elif feature_name == "SibSp":
        num = int(threshold)
        if direction == "right":
            if num == 0:
                return "Has siblings/spouse", "Has siblings or spouse aboard"
            elif num == 1:
                return "2+ siblings/spouses", "2 or more siblings/spouses"
            else:
                return f"{num}+ siblings/spouses", f"{num} or more siblings/spouses"
        else:
            if num == 1:
                return "No siblings/spouse", "No siblings or spouse aboard"
            else:
                return f"< {num} siblings/spouses", f"Fewer than {num} siblings/spouses"

    # Handle Parch (number of parents/children)
    elif feature_name == "Parch":
        num = int(threshold)
        if direction == "right":
            if num == 0:
                return "Has parents/children", "Has parents or children aboard"
            elif num == 1:
                return "2+ parents/children", "2 or more parents/children"
            else:
                return f"{num}+ parents/children", f"{num} or more parents/children"
        else:
            if num == 1:
                return "No parents/children", "No parents or children aboard"
            else:
                return f"< {num} parents/children", f"Fewer than {num} parents/children"

    # Handle Embarked (0.0=S, 1.0=C, 2.0=Q)
    elif feature_name == "Embarked":
        if direction == "right":
            if threshold <= 0.5:
                return "Embarked S", "Embarked at Southampton"
            elif threshold <= 1.5:
                return "Embarked C/Q", "Embarked at Cherbourg or Queenstown"
            else:
                return "Embarked Q", "Embarked at Queenstown"
        else:
            if threshold <= 0.5:
                return "Not S", "Did not embark at Southampton"
            elif threshold <= 1.5:
                return "Embarked S", "Embarked at Southampton"
            else:
                return "Embarked S/C", "Embarked at Southampton or Cherbourg"

    # Fallback for unknown features
    operator = ">=" if direction == "right" else "<"
    return f"{feature_name} {operator} {threshold:.2f}", f"{feature_name} {operator} {threshold:.2f}"


def _humanize_leaf(prediction: float, sample_count: int, survived_count: int) -> tuple[str, str]:
    """Convert leaf prediction into human-readable labels.

    Args:
        prediction: Survival probability (0.0 to 1.0).
        sample_count: Total number of samples in this leaf.
        survived_count: Number of passengers who survived.

    Returns:
        Tuple of (short_label, detailed_label) for display.
    """
    percentage = int(prediction * 100)
    short = f"Survival: {percentage}%"
    detailed = f"Survival rate: {percentage}% ({survived_count} of {sample_count} passengers survived)"
    return short, detailed


def _survival_color_category(prediction: float | None, is_leaf: bool) -> str:
    """Determine color category based on survival probability.

    Args:
        prediction: Survival probability (0.0 to 1.0) or None for split nodes.
        is_leaf: Whether this is a leaf node.

    Returns:
        Color category string for use in visualization.
    """
    if not is_leaf or prediction is None:
        return "split"

    if prediction > 0.7:
        return "high_survival"
    elif prediction >= 0.3:
        return "medium_survival"
    else:
        return "low_survival"


def _count_samples_in_tree(node, X_train: List, node_id: int = 0) -> dict[int, tuple[int, int]]:
    """Count samples reaching each node in the tree.

    Args:
        node: Root node of the decision tree.
        X_train: Training feature data.
        node_id: Current node ID (used for recursion).

    Returns:
        Dictionary mapping node_id to (total_samples, survived_samples).
    """
    from collections import Counter

    def walk_with_data(node, data, node_id, counter_gen):
        """Recursively walk tree with data to count samples."""
        current_id = next(counter_gen)

        # Count samples at this node
        total = len(data)

        # If leaf node, calculate survived based on prediction
        if node.feature_index is None or node.left is None or node.right is None:
            # For leaf nodes, prediction is the survival probability
            # Approximate survived count
            survived = int(node.prediction * total) if node.prediction is not None else 0
            result = {current_id: (total, survived)}
            return result, current_id

        # For split nodes, partition data
        left_data = []
        right_data = []
        for row in data:
            if row[node.feature_index] < node.threshold:
                left_data.append(row)
            else:
                right_data.append(row)

        # Set prediction as None for split nodes (we'll use average of children)
        result = {current_id: (total, 0)}  # Will be updated after processing children

        # Recursively process children
        left_counts, left_id = walk_with_data(node.left, left_data, node_id, counter_gen)
        right_counts, right_id = walk_with_data(node.right, right_data, node_id, counter_gen)

        result.update(left_counts)
        result.update(right_counts)

        return result, current_id

    def counter_gen():
        """Generate incrementing node IDs."""
        i = 0
        while True:
            yield i
            i += 1

    counts, _ = walk_with_data(node, X_train, node_id, counter_gen())
    return counts


def tree_structure_chart(
    tree,
    feature_names: List[str],
    X_train: List | None = None,
    y_train: List[int] | None = None,
    title: str | None = None
):
    """Visualize a decision tree with human-readable labels and survival coloring.

    Args:
        tree: Root node of the trained decision tree.
        feature_names: Headers describing each feature index in the tree.
        X_train: Optional training data for calculating sample counts.
        y_train: Optional training labels for calculating survival counts.
        title: Optional chart title.

    Returns:
        An Altair chart layering edges, nodes, and labels for the tree.
    """

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # Calculate sample counts if training data provided
    sample_counts = None
    if X_train is not None and y_train is not None:
        sample_counts = _count_samples_with_labels(tree, X_train, y_train)

    def walk(node, parent_id: int | None, depth: int, counter: Iterable[int], is_right_branch: bool = False):
        """Traverse the tree to populate node and edge collections."""

        node_id = next(counter)
        is_leaf = node.feature_index is None or node.left is None or node.right is None

        # Get sample counts for this node
        total_samples = 0
        survived_samples = 0
        if sample_counts is not None and node_id in sample_counts:
            total_samples, survived_samples = sample_counts[node_id]

        # Generate human-readable labels
        if is_leaf:
            # Leaf node - show survival rate
            prediction = node.prediction if node.prediction is not None else 0.0
            if total_samples > 0:
                short_label, detailed_label = _humanize_leaf(prediction, total_samples, survived_samples)
            else:
                # Fallback if no sample counts
                percentage = int(prediction * 100)
                short_label = f"Survival: {percentage}%"
                detailed_label = f"Survival rate: {percentage}%"
        else:
            # Split node - show demographic split
            feature_name = feature_names[node.feature_index]
            threshold = node.threshold
            # Determine which direction this node represents from parent
            direction = "right" if is_right_branch else "left"
            short_label, detailed_label = _humanize_split(feature_name, threshold, direction)

        # Determine color category
        prediction = node.prediction if is_leaf else None
        color_cat = _survival_color_category(prediction, is_leaf)

        # Add node data
        node_data = {
            "id": node_id,
            "parent": parent_id,
            "depth": depth,
            "label": short_label,
            "detailed_label": detailed_label,
            "y": len(nodes),
            "is_leaf": is_leaf,
            "color_category": color_cat,
            "samples": total_samples,
            "survived": survived_samples,
            "prediction": prediction if is_leaf else None
        }
        nodes.append(node_data)

        if node.left is None or node.right is None:
            return node_id

        # Recursively walk left and right branches
        left_id = walk(node.left, node_id, depth + 1, counter, is_right_branch=False)
        right_id = walk(node.right, node_id, depth + 1, counter, is_right_branch=True)

        # Add edges with labels
        edges.extend([
            {
                "source": node_id,
                "target": left_id,
                "x": depth,
                "x2": depth + 1,
                "y": nodes[node_id]["y"],
                "y2": nodes[left_id]["y"],
                "label": "No"
            },
            {
                "source": node_id,
                "target": right_id,
                "x": depth,
                "x2": depth + 1,
                "y": nodes[node_id]["y"],
                "y2": nodes[right_id]["y"],
                "label": "Yes"
            }
        ])
        return node_id

    def counter():
        """Generate incrementing integers for node identifiers."""
        i = 0
        while True:
            yield i
            i += 1

    walk(tree, None, 0, counter())

    # Define color scale for survival-based coloring
    color_scale = alt.Scale(
        domain=["high_survival", "medium_survival", "low_survival", "split"],
        range=["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]
    )

    # Create edge chart
    edge_chart = alt.Chart({"values": edges}).mark_line(color="#999", strokeWidth=2).encode(
        x="x:Q",
        x2="x2:Q",
        y="y:Q",
        y2="y2:Q"
    )

    # Create node chart with survival-based coloring
    node_chart = alt.Chart({"values": nodes}).mark_circle(size=200, opacity=0.9).encode(
        x=alt.X("depth:Q", title="Decision Depth"),
        y=alt.Y("y:Q", title="", axis=None),
        color=alt.Color(
            "color_category:N",
            scale=color_scale,
            legend=alt.Legend(
                title="Outcome",
                labelExpr="datum.label === 'high_survival' ? 'High Survival (>70%)' : datum.label === 'medium_survival' ? 'Medium Survival (30-70%)' : datum.label === 'low_survival' ? 'Low Survival (<30%)' : 'Decision Point'"
            )
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Split"),
            alt.Tooltip("detailed_label:N", title="Description"),
            alt.Tooltip("samples:Q", title="Passengers"),
            alt.Tooltip("survived:Q", title="Survived")
        ]
    )

    # Create label chart with concise labels
    label_chart = alt.Chart({"values": nodes}).mark_text(
        align="left",
        dx=8,
        fontSize=11,
        fontWeight="bold"
    ).encode(
        x="depth:Q",
        y="y:Q",
        text="label:N"
    )

    # Combine charts
    combined = edge_chart + node_chart + label_chart

    # Add title if provided
    if title:
        combined = combined.properties(
            title=title,
            width=600,
            height=400
        )
    else:
        combined = combined.properties(
            title="Decision Tree: Demographic Survival Analysis",
            width=600,
            height=400
        )

    return combined


def _count_samples_with_labels(node, X_train: List, y_train: List[int]) -> dict[int, tuple[int, int]]:
    """Count samples and survivors reaching each node in the tree.

    Args:
        node: Root node of the decision tree.
        X_train: Training feature data.
        y_train: Training labels (0=died, 1=survived).

    Returns:
        Dictionary mapping node_id to (total_samples, survived_samples).
    """
    def walk_with_data(node, data, labels, counter_gen):
        """Recursively walk tree with data to count samples."""
        current_id = next(counter_gen)

        # Count samples at this node
        total = len(data)
        survived = sum(labels)

        # If leaf node, return counts
        if node.feature_index is None or node.left is None or node.right is None:
            return {current_id: (total, survived)}, current_id

        # For split nodes, partition data
        left_data = []
        left_labels = []
        right_data = []
        right_labels = []

        for row, label in zip(data, labels):
            if row[node.feature_index] < node.threshold:
                left_data.append(row)
                left_labels.append(label)
            else:
                right_data.append(row)
                right_labels.append(label)

        # Store counts for this split node
        result = {current_id: (total, survived)}

        # Recursively process children
        left_counts, _ = walk_with_data(node.left, left_data, left_labels, counter_gen)
        right_counts, _ = walk_with_data(node.right, right_data, right_labels, counter_gen)

        result.update(left_counts)
        result.update(right_counts)

        return result, current_id

    def counter_gen():
        """Generate incrementing node IDs."""
        i = 0
        while True:
            yield i
            i += 1

    counts, _ = walk_with_data(node, X_train, y_train, counter_gen())
    return counts


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
