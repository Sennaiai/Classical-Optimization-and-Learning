import unittest, tempfile
from pathlib import Path
from copy import deepcopy
import numpy as np
from classical_studies.baselines.molecular import (
    fit_classical,
    predict,
    evaluate,
    write_model,
    read_model,
    validate_model,
)


def fixture():
    train = [f"train_{i}" for i in range(12)]
    val = [f"validation_{i}" for i in range(6)]
    test = [f"test_{i}" for i in range(6)]
    parts = {"train": train, "validation": val, "test": test}
    groups = {i: i for v in parts.values() for i in v}
    x = np.array([[i % 2, 2 * (i % 2), i % 3] for i in range(12)])
    v = np.array([[i % 2, 2 * (i % 2), i % 3] for i in range(6)])
    y = {i: j % 2 for j, i in enumerate(train)}
    vy = {i: j % 2 for j, i in enumerate(val)}
    return x, v, y, vy, parts, groups


def fitted():
    x, v, y, vy, p, g = fixture()
    return fit_classical(
        x,
        y,
        validation_X=v,
        validation_labels=vy,
        partitions=p,
        groups=g,
        feature_names=["count0", "count1", "count2"],
    )


class MolecularTests(unittest.TestCase):
    def test_selected_scaled_counts_fit_and_enforce_their_domain(self):
        x, v, y, vy, parts, groups = fixture()
        model = fit_classical(
            x / 2,
            y,
            validation_X=v / 2,
            validation_labels=vy,
            partitions=parts,
            groups=groups,
            feature_names=["a", "b", "c"],
            representation="selected_counts",
        )
        p = predict(
            model,
            v / 2,
            feature_names=model["feature_names"],
            preprocessing_id=model["preprocessing_id"],
        )
        self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError):
            predict(
                model,
                v + 2,
                feature_names=model["feature_names"],
                preprocessing_id=model["preprocessing_id"],
            )

    def test_train_only_scaler_and_portable_model_roundtrip(self):
        x, v, y, vy, parts, groups = fixture()
        model = fitted()
        np.testing.assert_allclose(model["scaler_mean"], x.mean(axis=0))
        self.assertFalse(model["test_labels_used_in_fit"])
        p = predict(
            model,
            v,
            feature_names=model["feature_names"],
            preprocessing_id=model["preprocessing_id"],
        )
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            write_model(path, model)
            restored = read_model(path)
            np.testing.assert_array_equal(
                p,
                predict(
                    restored,
                    v,
                    feature_names=model["feature_names"],
                    preprocessing_id=model["preprocessing_id"],
                ),
            )

    def test_identity_and_scaffold_leakage_rejected(self):
        x, v, y, vy, p, g = fixture()
        g[p["validation"][0]] = g[p["train"][0]]
        with self.assertRaises(ValueError):
            fit_classical(
                x,
                y,
                validation_X=v,
                validation_labels=vy,
                partitions=p,
                groups=g,
                feature_names=["a", "b", "c"],
            )
        x, v, y, vy, p, g = fixture()
        p["test"][0] = p["train"][0]
        with self.assertRaises(ValueError):
            fit_classical(
                x,
                y,
                validation_X=v,
                validation_labels=vy,
                partitions=p,
                groups=g,
                feature_names=["a", "b", "c"],
            )

    def test_schema_tampering_and_feature_permutations_rejected(self):
        model = fitted()
        x, v, *_ = fixture()
        bad = deepcopy(model)
        bad["coefficient"][0] += 1
        with self.assertRaises(ValueError):
            validate_model(bad)
        with self.assertRaises(ValueError):
            predict(
                model,
                v,
                feature_names=["count1", "count0", "count2"],
                preprocessing_id=model["preprocessing_id"],
            )
        with self.assertRaises(ValueError):
            predict(
                model,
                np.full(v.shape, np.nan),
                feature_names=model["feature_names"],
                preprocessing_id=model["preprocessing_id"],
            )

    def test_complete_frozen_test_partition_required(self):
        model = fitted()
        _, v, _, _, parts, _ = fixture()
        labels = {i: j % 2 for j, i in enumerate(parts["test"])}
        r = evaluate(
            model,
            v,
            labels,
            ids=parts["test"],
            feature_names=model["feature_names"],
            preprocessing_id=model["preprocessing_id"],
        )
        self.assertEqual(r["samples"], 6)
        self.assertGreater(r["roc_auc"], 0.9)
        with self.assertRaises(ValueError):
            evaluate(
                model,
                v,
                labels,
                ids=parts["test"][:-1],
                feature_names=model["feature_names"],
                preprocessing_id=model["preprocessing_id"],
            )


if __name__ == "__main__":
    unittest.main()
