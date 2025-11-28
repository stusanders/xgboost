"""Data utilities for the Titanic workshop.

This module keeps the workshop fully offline by bundling a small sample of the
Titanic passenger manifest. The helper functions create an ``input/`` directory
if needed, ensure the CSV exists (either by download or fallback), and expose
simple loading and feature/label splitting helpers without external
dependencies.
"""
from __future__ import annotations

import csv
import pathlib
from typing import Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# A lightweight fallback dataset for offline execution. The rows include the
# columns required by the workshop: Survived, Pclass, Sex, Age, SibSp, Parch,
# Fare, Embarked.
_FALLBACK_ROWS: list[dict[str, str]] = [
    {
        "Survived": "0",
        "Pclass": "3",
        "Sex": "male",
        "Age": "22.0",
        "SibSp": "1",
        "Parch": "0",
        "Fare": "7.25",
        "Embarked": "S",
    },
    {
        "Survived": "1",
        "Pclass": "1",
        "Sex": "female",
        "Age": "38.0",
        "SibSp": "1",
        "Parch": "0",
        "Fare": "71.2833",
        "Embarked": "C",
    },
    {
        "Survived": "1",
        "Pclass": "3",
        "Sex": "female",
        "Age": "26.0",
        "SibSp": "0",
        "Parch": "0",
        "Fare": "7.925",
        "Embarked": "S",
    },
    {
        "Survived": "1",
        "Pclass": "1",
        "Sex": "female",
        "Age": "35.0",
        "SibSp": "1",
        "Parch": "0",
        "Fare": "53.1",
        "Embarked": "S",
    },
    {
        "Survived": "0",
        "Pclass": "3",
        "Sex": "male",
        "Age": "35.0",
        "SibSp": "0",
        "Parch": "0",
        "Fare": "8.05",
        "Embarked": "S",
    },
    {
        "Survived": "0",
        "Pclass": "3",
        "Sex": "male",
        "Age": "28.0",
        "SibSp": "0",
        "Parch": "0",
        "Fare": "8.4583",
        "Embarked": "Q",
    },
    {
        "Survived": "0",
        "Pclass": "1",
        "Sex": "male",
        "Age": "54.0",
        "SibSp": "0",
        "Parch": "0",
        "Fare": "51.8625",
        "Embarked": "S",
    },
    {
        "Survived": "0",
        "Pclass": "3",
        "Sex": "male",
        "Age": "2.0",
        "SibSp": "3",
        "Parch": "1",
        "Fare": "21.075",
        "Embarked": "S",
    },
    {
        "Survived": "1",
        "Pclass": "3",
        "Sex": "female",
        "Age": "27.0",
        "SibSp": "0",
        "Parch": "2",
        "Fare": "11.1333",
        "Embarked": "S",
    },
    {
        "Survived": "1",
        "Pclass": "2",
        "Sex": "female",
        "Age": "14.0",
        "SibSp": "1",
        "Parch": "0",
        "Fare": "30.0708",
        "Embarked": "C",
    },
]


def _write_rows(csv_path: pathlib.Path, rows: Iterable[dict[str, str]]) -> None:
    """Write iterable rows to ``csv_path`` with consistent headers."""

    headers = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_data(data_dir: pathlib.Path | str = "input") -> pathlib.Path:
    """Ensure the Titanic CSV exists locally, downloading it if possible.

    Args:
        data_dir: Directory where the dataset should be stored. The directory is
            created if it does not already exist.

    Returns:
        Path to the downloaded or fallback CSV file on disk.
    """

    data_path = pathlib.Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / "titanic.csv"

    if csv_path.exists():
        return csv_path

    try:
        with urlopen(DATA_URL, timeout=10) as response:  # type: ignore[arg-type]
            content = response.read().decode("utf-8")
        csv_path.write_text(content, encoding="utf-8")
    except (URLError, TimeoutError, OSError):
        _write_rows(csv_path, _FALLBACK_ROWS)
    return csv_path


def load_dataset(path: pathlib.Path | str) -> List[dict[str, str]]:
    """Load the Titanic dataset from disk into a list of dictionaries."""

    with pathlib.Path(path).open() as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def split_features_labels(
    rows: List[dict[str, str]],
) -> Tuple[List[dict[str, str]], list[int]]:
    """Split raw Titanic rows into feature dictionaries and integer labels."""

    labels: list[int] = []
    features: list[dict[str, str]] = []
    for row in rows:
        labels.append(int(row["Survived"]))
        features.append(
            {
                "Pclass": row.get("Pclass", ""),
                "Sex": row.get("Sex", ""),
                "Age": row.get("Age", ""),
                "SibSp": row.get("SibSp", ""),
                "Parch": row.get("Parch", ""),
                "Fare": row.get("Fare", ""),
                "Embarked": row.get("Embarked", ""),
            }
        )
    return features, labels
