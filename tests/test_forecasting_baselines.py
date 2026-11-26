import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from classical_studies.baselines.forecasting.data_contract import (
    GRID,
    prepare,
    settlement_start,
    load_data,
)
from classical_studies.baselines.forecasting import (
    fit_forecaster,
    select_model,
    echo_features,
    model_factory,
)
from classical_studies.baselines.forecasting.models import metrics


def synthetic_data():
    rng = np.random.default_rng(38)
    n = 1100
    values = rng.normal(size=(n, 7))
    values[:, 0] = 1000 + 10 * np.sin(np.arange(n) / 20) + rng.normal(size=n)
    return prepare(
        pd.DataFrame(
            values,
            index=pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"),
            columns=GRID,
        )
    )


class ForecastTests(unittest.TestCase):
    def test_metrics_reject_empty_or_nonfinite_evidence(self):
        for truth, pred, scale in [
            ([], [], 1),
            ([np.nan], [0], 1),
            ([0], [np.inf], 1),
            ([0], [0], np.nan),
        ]:
            with self.assertRaises(ValueError):
                metrics(truth, pred, scale)

    def test_causal_features_and_chronological_folds(self):
        d = synthetic_data()
        self.assertTrue(np.all(d.source_times < d.times))
        for tr, va in d.folds():
            self.assertLess(tr[-1], va[0])
            self.assertLess(va[-1], d.test_start)
        self.assertEqual(d.raw.shape[1], 109)

    def test_test_labels_cannot_change_parameter_selection(self):
        a = synthetic_data()
        b = deepcopy(a)
        b.y[b.test_start :] += 1e9
        grid = [{"alpha": 0.1}, {"alpha": 10.0}]
        self.assertEqual(
            select_model("ridge", grid, a.raw, a), select_model("ridge", grid, b.raw, b)
        )

    def test_models_are_independently_fit_and_callable(self):
        d = synthetic_data()
        for family, grid in [
            ("ridge", [{"alpha": 10.0}]),
            ("rf", [{"max_depth": 3, "min_samples_leaf": 8}]),
            ("boost", [{"max_leaf_nodes": 7, "l2_regularization": 1.0}]),
        ]:
            model = fit_forecaster(family, grid, d)
            pred = model.predict(d.raw[d.test_start :], d.persistence[d.test_start :])
            self.assertEqual(pred.shape, d.y[d.test_start :].shape)
            self.assertTrue(np.isfinite(pred).all())
            with self.assertRaises(ValueError):
                model.predict(np.zeros((2, 1)), np.zeros(2))

    def test_echo_is_repeatable_finite_and_dimension_matched(self):
        d = synthetic_data()
        x = d.windows[:20]
        a = echo_features(x, 11)
        b = echo_features(x, 11)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (20, 33))
        self.assertTrue(np.isfinite(a).all())
        with self.assertRaises(ValueError):
            echo_features(np.full((2, 8, 7), np.nan), 11)

    def test_public_data_checksum_and_dst_contract(self):
        frame = load_data()
        self.assertGreater(len(frame), 1000)
        with self.assertRaises(ValueError):
            settlement_start("2025-03-30", 47)
        self.assertEqual(
            (
                settlement_start("2025-10-26", 50) - settlement_start("2025-10-26", 1)
            ).total_seconds(),
            49 * 1800,
        )


if __name__ == "__main__":
    unittest.main()
