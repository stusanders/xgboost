"""Smoke tests for the Titanic workshop CLI helpers."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from titanic_workshop.main import run_experiments


class RunExperimentsTest(unittest.TestCase):
    """Verify that workshop experiments execute for all model types."""

    def test_all_models_execute_and_return_metrics(self) -> None:
        """All models should run end-to-end and produce bounded metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_experiments(
                ["tree", "forest", "xgboost"],
                data_dir=pathlib.Path(tmpdir),
                max_depth=2,
                forest_trees=3,
                visualize=True,
            )

        self.assertEqual(len(results), 3)
        for result in results:
            with self.subTest(model=result.name):
                self.assertGreaterEqual(result.accuracy, 0.0)
                self.assertLessEqual(result.accuracy, 1.0)
                self.assertGreaterEqual(result.roc_auc, 0.0)
                self.assertLessEqual(result.roc_auc, 1.0)
                # Visualizations should be present when requested
                self.assertTrue(result.visualizations)


if __name__ == "__main__":
    unittest.main()
