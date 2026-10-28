"""Scientific boundary cases and fixed-snapshot evidence contracts."""

from pathlib import Path
import copy
import json
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling import design, fit, source, spline_values
from evaluate import metrics, ordered_names, ranking_comparison


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(
            (ROOT / "results/models/final/additive_full.json").read_text(
                encoding="utf-8"
            )
        )

    def test_held_year_is_rejected_by_fitting(self):
        with self.assertRaisesRegex(AssertionError, "held-year"):
            fit(source(2012).head(3), {"family": "linear", "lambda": 0.01})

    def test_target_components_and_id_cannot_change_design(self):
        sample = source(2011).head(4)
        altered = sample.copy()
        altered[["cnt", "casual", "registered", "instant"]] = 999999
        np.testing.assert_array_equal(
            design(sample, self.model["transform"]),
            design(altered, self.model["transform"]),
        )

    def test_inference_does_not_refit_training_transform(self):
        transform = copy.deepcopy(self.model["transform"])
        before = json.dumps(transform, sort_keys=True)
        sample = source(2012).head(5).copy()
        sample["temp"] = 1.4
        assert np.isfinite(design(sample, transform)).all()
        self.assertEqual(before, json.dumps(transform, sort_keys=True))

    def test_spline_extrapolation_is_linear_at_boundary(self):
        spec = self.model["transform"]["splines"]["temp"]
        edge = spec["t"][-4]
        values = spline_values(np.array([edge, edge + 0.1, edge + 0.2]), spec)
        np.testing.assert_allclose(
            values[2] - values[1], values[1] - values[0], atol=1e-12
        )

    def test_mse_is_squared_count_error_after_declared_clipping(self):
        result = metrics(np.array([0.0, 1.0]), np.array([-3.0, 3.0]), 4.0)
        self.assertEqual(result["mse"], 2.0)
        self.assertEqual(result["normalized_mse"], 0.5)
        self.assertEqual(result["negative_raw_predictions"], 1)

    def test_rank_ties_are_alphabetical(self):
        self.assertEqual(
            ordered_names({"temp": 0, "atemp": 0, "weather": -1}),
            ["atemp", "temp", "weather"],
        )
        self.assertEqual(
            ranking_comparison(
                {"a": 3, "b": 2, "c": 1, "d": 0}, {"a": 0, "b": 1, "c": 2, "d": 3}
            )["spearman"],
            -1,
        )

    def test_negative_observed_importance_is_retained(self):
        rows = pd.read_csv(ROOT / "results/importance.csv")
        self.assertLess(rows.importance.min(), 0)

    def test_conditioning_context_is_structurally_unchanged(self):
        manifest = json.loads(
            (ROOT / "results/permutations.json").read_text(encoding="utf-8")
        )
        for group in ["hour", "month", "workingday"]:
            self.assertEqual(manifest["unchanged_row_fractions"][group], [1.0] * 5)
        self.assertEqual(
            manifest["raw_columns_by_group"]["joint_temperature"], ["temp", "atemp"]
        )


if __name__ == "__main__":
    unittest.main()
