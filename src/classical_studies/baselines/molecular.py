"""Classical-only portable logistic baselines extracted from the ADMET controls.

Input matrices are full Morgan counts or previously selected/scaled counts.
Scaffold group keys must be supplied by the caller; no quantum record is used.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, math, numbers, re, warnings
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_CELLS = 30_000_000
MAX_ROWS = 50_000
MAX_FEATURES = 65_536


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ReadoutConfig:
    regularization_candidates: tuple[float, ...] = (0.1, 1.0, 10.0)
    class_weight: str | None = None
    max_iterations: int = 5000
    tolerance: float = 1e-9

    def validate(self):
        values = self.regularization_candidates
        if (
            not isinstance(values, (tuple, list))
            or not 1 <= len(values) <= 16
            or any(
                isinstance(x, bool)
                or not isinstance(x, numbers.Real)
                or not math.isfinite(x)
                or x <= 0
                for x in values
            )
        ):
            raise ValueError(
                "Regularization candidates must be bounded positive finite real values"
            )
        if list(values) != sorted(set(values)):
            raise ValueError("Regularization candidates must be unique and increasing")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or balanced")
        for name, minimum, maximum in (("max_iterations", 10, 100000),):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} is outside its integer budget")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, numbers.Real)
            or not math.isfinite(self.tolerance)
            or not 1e-12 <= self.tolerance <= 1e-3
        ):
            raise ValueError("tolerance must lie in [1e-12, 1e-3]")


def _configuration(config):
    config = ReadoutConfig() if config is None else config
    if not isinstance(config, ReadoutConfig):
        raise ValueError("config must be a ReadoutConfig")
    config.validate()
    return config


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _ids(values, name):
    if (
        not isinstance(values, (list, tuple))
        or not 1 <= len(values) <= MAX_ROWS
        or any(not isinstance(x, str) or not x or x != x.strip() for x in values)
    ):
        raise ValueError(f"{name} must be nonempty molecule identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate molecule identifiers")
    return list(values)


def _hash(value, name):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identifier")
    return value


def _matrix(values, name):
    if isinstance(values, np.ndarray):
        shape = values.shape
        if values.ndim != 2:
            raise ValueError(f"{name} must be a rectangular numeric matrix")
    elif (
        isinstance(values, (list, tuple))
        and values
        and isinstance(values[0], (list, tuple, np.ndarray))
    ):
        shape = (len(values), len(values[0]))
        if any(
            not isinstance(row, (list, tuple, np.ndarray)) or len(row) != shape[1]
            for row in values
        ):
            raise ValueError(f"{name} must be rectangular")
    else:
        raise ValueError(f"{name} must be a rectangular numeric matrix")
    if (
        not 1 <= shape[0] <= MAX_ROWS
        or not 1 <= shape[1] <= MAX_FEATURES
        or shape[0] * shape[1] > MAX_CELLS
    ):
        raise ValueError(f"{name} exceeds the bounded readout allocation")
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _labels(labels, ids, name):
    if not isinstance(labels, dict) or set(labels) != set(ids):
        raise ValueError(f"{name} must map exactly the supplied molecule IDs to labels")
    values = [labels[i] for i in ids]
    if any(
        isinstance(x, complex)
        or not isinstance(x, (numbers.Real, np.bool_))
        or not math.isfinite(x)
        or x not in (0, 1)
        for x in values
    ):
        raise ValueError(f"{name} must contain binary real labels")
    result = np.asarray(values, dtype=int)
    if set(result.tolist()) != {0, 1}:
        raise ValueError(f"{name} requires both binary classes")
    return result


def binary_metrics(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or len(y) < 2
        or y.dtype.kind not in "biuf"
        or np.iscomplexobj(y)
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or set(y.tolist()) != {0, 1}
        or p.dtype.kind not in "biuf"
        or np.iscomplexobj(p)
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(
            "Metrics require aligned finite binary labels and probabilities with both classes"
        )
    prediction = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "samples": len(y),
        "positive_prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, prediction)),
        "confusion_at_0_5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _feature_names(names, width):
    if (
        not isinstance(names, (list, tuple))
        or len(names) != width
        or any(not isinstance(x, str) or not x for x in names)
        or len(set(names)) != width
    ):
        raise ValueError("Feature names must uniquely describe every matrix column")
    return list(names)


def _fit_numeric(
    train_X,
    train_labels,
    train_ids,
    validation_X,
    validation_labels,
    validation_ids,
    *,
    feature_names,
    feature_schema_id,
    endpoint,
    preprocessing_id,
    split_id,
    partitions,
    source,
    kind,
    config,
):
    config.validate()
    train_X, validation_X = (
        _matrix(train_X, "training features"),
        _matrix(validation_X, "validation features"),
    )
    train_ids, validation_ids = (
        _ids(train_ids, "training IDs"),
        _ids(validation_ids, "validation IDs"),
    )
    if (
        len(train_ids) != len(train_X)
        or len(validation_ids) != len(validation_X)
        or set(train_ids) != set(partitions["train"])
        or set(validation_ids) != set(partitions["validation"])
    ):
        raise ValueError(
            "Feature records must completely cover their declared training/validation partitions"
        )
    if train_X.shape[1] != validation_X.shape[1]:
        raise ValueError("Training and validation feature dimensions differ")
    names = _feature_names(feature_names, train_X.shape[1])
    y_train, y_validation = (
        _labels(train_labels, train_ids, "training labels"),
        _labels(validation_labels, validation_ids, "validation labels"),
    )
    if min(np.bincount(y_train, minlength=2)) < 2:
        raise ValueError("Training requires at least two molecules in each class")
    scaler = StandardScaler().fit(train_X)
    transformed, validation_transformed = (
        scaler.transform(train_X),
        scaler.transform(validation_X),
    )
    trials, best = [], None
    for C in config.regularization_candidates:
        classifier = LogisticRegression(
            C=float(C),
            class_weight=config.class_weight,
            solver="lbfgs",
            max_iter=config.max_iterations,
            tol=config.tolerance,
            random_state=42,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(transformed, y_train)
        except ConvergenceWarning:
            raise ValueError(
                "Logistic readout did not converge; model was not accepted"
            ) from None
        metrics = binary_metrics(
            y_validation, classifier.predict_proba(validation_transformed)[:, 1]
        )
        trials.append({"C": float(C), "metrics": metrics})
        # Stable tie policy: sorted candidates retain the smaller C.
        if best is None or metrics["roc_auc"] > best[0]:
            best = (metrics["roc_auc"], classifier, float(C))
    classifier = best[1]
    coefficients = classifier.coef_[0].astype(float)
    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(classifier.intercept_).all()
    ):
        raise ValueError("Fitted model contains nonfinite coefficients")
    artifact = {
        "model_schema_version": 1,
        "status": "fitted",
        "model_type": "standardized_binary_logistic",
        "kind": kind,
        "endpoint": endpoint,
        "preprocessing_id": preprocessing_id,
        "split_id": split_id,
        "feature_schema_id": _hash(feature_schema_id, "feature_schema_id"),
        "feature_names": names,
        "data_source": source,
        "classes": [0, 1],
        "threshold": 0.5,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": coefficients.tolist(),
        "intercept": float(classifier.intercept_[0]),
        "selected_C": best[2],
        "config": json.loads(canonical_json(asdict(config))),
        "train_molecule_ids": train_ids,
        "validation_molecule_ids": validation_ids,
        "test_molecule_ids": list(partitions["test"]),
        "training_label_sha256": _digest(
            [[i, int(train_labels[i])] for i in sorted(train_ids)]
        ),
        "validation_label_sha256": _digest(
            [[i, int(validation_labels[i])] for i in sorted(validation_ids)]
        ),
        "training_class_counts": {str(i): int(np.sum(y_train == i)) for i in (0, 1)},
        "training_positive_prevalence": float(np.mean(y_train)),
        "validation_selection": {
            "metric": "roc_auc",
            "tie_policy": "smaller C",
            "trials": trials,
        },
        "probability_policy": "sigmoid readout; no separate calibration or predictive interval claim",
        "test_labels_used_in_fit": False,
    }
    artifact["model_id"] = _digest(artifact)
    validate_model(artifact)
    # Verify the exported arithmetic against the fitted sklearn estimator.
    exported = _numeric_predict(artifact, validation_X)
    if not np.allclose(
        exported,
        classifier.predict_proba(validation_transformed)[:, 1],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(
            "Portable logistic arithmetic differs from the fitted estimator"
        )
    return artifact


def validate_model(model):
    if (
        not isinstance(model, dict)
        or type(model.get("model_schema_version")) is not int
        or model["model_schema_version"] != 1
    ):
        raise ValueError("Unsupported readout model schema")
    expected = {
        "model_schema_version",
        "status",
        "model_type",
        "kind",
        "endpoint",
        "preprocessing_id",
        "split_id",
        "feature_schema_id",
        "feature_names",
        "data_source",
        "classes",
        "threshold",
        "scaler_mean",
        "scaler_scale",
        "coefficient",
        "intercept",
        "selected_C",
        "config",
        "train_molecule_ids",
        "validation_molecule_ids",
        "test_molecule_ids",
        "training_label_sha256",
        "validation_label_sha256",
        "training_class_counts",
        "training_positive_prevalence",
        "validation_selection",
        "probability_policy",
        "test_labels_used_in_fit",
        "model_id",
    }
    if set(model) != expected:
        raise ValueError("Readout model fields differ from schema")
    body = {key: value for key, value in model.items() if key != "model_id"}
    if _hash(model.get("model_id"), "model_id") != _digest(body):
        raise ValueError("Readout model checksum differs from its content")
    if (
        model.get("status") != "fitted"
        or model.get("model_type") != "standardized_binary_logistic"
        or model.get("kind")
        not in {"classical_full_counts", "classical_selected_counts"}
        or model.get("classes") != [0, 1]
        or any(type(x) is not int for x in model.get("classes", []))
        or model.get("threshold") != 0.5
        or model.get("test_labels_used_in_fit") is not False
    ):
        raise ValueError("Invalid readout model policy")
    for name in ("preprocessing_id", "split_id", "feature_schema_id"):
        _hash(model.get(name), name)
    if (
        not isinstance(model.get("endpoint"), str)
        or not model["endpoint"]
        or not isinstance(model.get("data_source"), str)
    ):
        raise ValueError("Invalid readout endpoint/source")
    names = model.get("feature_names")
    if not isinstance(names, list):
        raise ValueError("Model feature names must be a list")
    _feature_names(names, len(names))
    if not 1 <= len(names) <= MAX_FEATURES:
        raise ValueError("Readout feature budget exceeded")
    for name in ("scaler_mean", "scaler_scale", "coefficient"):
        values = model.get(name)
        if (
            not isinstance(values, list)
            or len(values) != len(names)
            or any(
                isinstance(x, bool)
                or not isinstance(x, numbers.Real)
                or not math.isfinite(x)
                for x in values
            )
        ):
            raise ValueError(f"Invalid model {name}")
    if any(value <= 0 for value in model["scaler_scale"]):
        raise ValueError("Model scales must be positive")
    if (
        isinstance(model.get("intercept"), bool)
        or not isinstance(model.get("intercept"), numbers.Real)
        or not math.isfinite(model["intercept"])
    ):
        raise ValueError("Model intercept must be finite")
    try:
        config = ReadoutConfig(**model["config"])
        config.validate()
    except (TypeError, ValueError):
        raise ValueError("Invalid model configuration") from None
    if (
        isinstance(model.get("selected_C"), bool)
        or not isinstance(model.get("selected_C"), numbers.Real)
        or model.get("selected_C") not in config.regularization_candidates
    ):
        raise ValueError("Selected C is absent from the declared search")
    sets = {
        key: set(_ids(model.get(key), key))
        for key in (
            "train_molecule_ids",
            "validation_molecule_ids",
            "test_molecule_ids",
        )
    }
    if any(
        sets[a] & sets[b]
        for a, b in (
            ("train_molecule_ids", "validation_molecule_ids"),
            ("train_molecule_ids", "test_molecule_ids"),
            ("validation_molecule_ids", "test_molecule_ids"),
        )
    ):
        raise ValueError("Readout model partitions overlap")
    counts = model["training_class_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"0", "1"}
        or any(type(x) is not int or x < 2 for x in counts.values())
        or sum(counts.values()) != len(sets["train_molecule_ids"])
    ):
        raise ValueError("Invalid model training class counts")
    prevalence = model["training_positive_prevalence"]
    if (
        isinstance(prevalence, bool)
        or not isinstance(prevalence, numbers.Real)
        or not math.isfinite(prevalence)
        or not math.isclose(
            prevalence, counts["1"] / sum(counts.values()), rel_tol=1e-12
        )
    ):
        raise ValueError("Training prevalence differs from class counts")
    _hash(model["training_label_sha256"], "training_label_sha256")
    _hash(model["validation_label_sha256"], "validation_label_sha256")
    selection = model["validation_selection"]
    if (
        not isinstance(selection, dict)
        or set(selection) != {"metric", "tie_policy", "trials"}
        or selection["metric"] != "roc_auc"
        or selection["tie_policy"] != "smaller C"
        or not isinstance(selection["trials"], list)
        or len(selection["trials"]) != len(config.regularization_candidates)
    ):
        raise ValueError("Invalid validation-only selection metadata")
    values = []
    for trial, C in zip(selection["trials"], config.regularization_candidates):
        if (
            not isinstance(trial, dict)
            or set(trial) != {"C", "metrics"}
            or trial["C"] != C
            or not isinstance(trial["metrics"], dict)
        ):
            raise ValueError(
                "Validation trial differs from the declared regularization search"
            )
        score = trial["metrics"].get("roc_auc")
        if (
            isinstance(score, bool)
            or not isinstance(score, numbers.Real)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Invalid validation ROC-AUC")
        values.append(score)
    if (
        model["selected_C"]
        != config.regularization_candidates[values.index(max(values))]
    ):
        raise ValueError("Selected C disagrees with validation-only ranking")
    sources = {"classical_molecular_counts"}
    if model["data_source"] not in sources:
        raise ValueError("Model evidence source is incompatible with its type")
    return True


def _numeric_predict(model, X):
    X = _matrix(X, "prediction features")
    if X.shape[1] != len(model["feature_names"]):
        raise ValueError("Prediction feature dimension differs from model")
    values = (X - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    scores = values @ np.asarray(model["coefficient"]) + model["intercept"]
    if not np.isfinite(scores).all():
        raise ValueError("Prediction arithmetic overflowed")
    return expit(scores)


def fit_classical(
    train_X,
    train_labels,
    *,
    validation_X,
    validation_labels,
    partitions,
    groups,
    feature_names,
    representation="full_counts",
    endpoint="molecular_property",
    config=None,
):
    config = _configuration(config)
    if not isinstance(partitions, dict) or set(partitions) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Declare train, validation and untouched-test IDs")
    ids = {name: _ids(partitions[name], name) for name in partitions}
    if any(
        set(ids[a]) & set(ids[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Molecular identities overlap partitions")
    all_ids = set().union(*(set(v) for v in ids.values()))
    if (
        not isinstance(groups, dict)
        or set(groups) != all_ids
        or any(not isinstance(v, str) or not v for v in groups.values())
    ):
        raise ValueError(
            "Provide a nonempty scaffold group for every declared molecule"
        )
    group_sets = {name: {groups[i] for i in values} for name, values in ids.items()}
    if any(
        group_sets[a] & group_sets[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Scaffold groups overlap partitions")
    if representation not in ("full_counts", "selected_counts"):
        raise ValueError("Unknown molecular representation")
    x = _matrix(train_X, "training features")
    v = _matrix(validation_X, "validation features")
    for a in (x, v):
        if (
            np.any(a < 0)
            or (representation == "full_counts" and not np.equal(a, np.floor(a)).all())
            or (representation == "selected_counts" and np.any(a > 1))
        ):
            raise ValueError("Features violate count/scaled-count domains")
    names = _feature_names(feature_names, x.shape[1])
    preprocessing_id = _digest(
        {
            "representation": representation,
            "feature_names": names,
            "train_ids": ids["train"],
            "train_features_sha256": hashlib.sha256(
                x.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
        }
    )
    return _fit_numeric(
        x,
        train_labels,
        ids["train"],
        v,
        validation_labels,
        ids["validation"],
        feature_names=names,
        feature_schema_id=_digest(
            {"preprocessing_id": preprocessing_id, "feature_names": names}
        ),
        endpoint=endpoint,
        preprocessing_id=preprocessing_id,
        split_id=_digest({"partitions": ids, "scaffold_groups": groups}),
        partitions=ids,
        source="classical_molecular_counts",
        kind="classical_" + representation,
        config=config,
    )


def predict(model, X, *, feature_names, preprocessing_id):
    validate_model(model)
    if (
        list(feature_names) != model["feature_names"]
        or preprocessing_id != model["preprocessing_id"]
    ):
        raise ValueError(
            "Prediction feature layout or preprocessing binding differs from fit"
        )
    values = _matrix(X, "prediction features")
    if (
        np.any(values < 0)
        or (
            model["kind"] == "classical_full_counts"
            and not np.equal(values, np.floor(values)).all()
        )
        or (model["kind"] == "classical_selected_counts" and np.any(values > 1))
    ):
        raise ValueError("Prediction features violate representation domains")
    return _numeric_predict(model, values)


def evaluate(model, X, labels, *, ids, feature_names, preprocessing_id):
    if set(_ids(ids, "test IDs")) != set(model["test_molecule_ids"]):
        raise ValueError("Evaluation must cover the frozen complete test partition")
    return binary_metrics(
        _labels(labels, ids, "test labels"),
        predict(
            model, X, feature_names=feature_names, preprocessing_id=preprocessing_id
        ),
    )


def write_model(path, model):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(canonical_json(model) + "\n")
    temp.replace(path)


def read_model(path):
    result = json.loads(Path(path).read_text())
    validate_model(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a portable classical logistic model from a JSON matrix/partition specification."
    )
    parser.add_argument("specification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.specification.read_text())
    if "config" in spec:
        spec["config"] = ReadoutConfig(**spec["config"])
    write_model(args.output, fit_classical(**spec))
