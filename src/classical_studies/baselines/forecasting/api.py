"""Fit and call the migrated causal classical forecasters without quantum code."""

from dataclasses import dataclass
import numpy as np
from .models import (
    model_factory,
    select_model,
    metrics,
    fit_evaluate,
    CLASSICAL,
    RIDGE_GRID,
)
from .data_contract import SEEDS, load_data, prepare, split_description
from .echo import echo_features


@dataclass
class FittedForecaster:
    model: object
    width: int
    selected_params: dict
    validation_candidates: list

    def predict(self, features, persistence):
        x = np.asarray(features, dtype=float)
        anchor = np.asarray(persistence, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != self.width
            or anchor.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(anchor).all()
        ):
            raise ValueError("Invalid causal feature/anchor matrix")
        return anchor + self.model.predict(x)


def fit_forecaster(kind, grid, data, features=None):
    x = data.raw if features is None else np.asarray(features, dtype=float)
    params, candidates = select_model(kind, grid, x, data)
    model = model_factory(kind, params).fit(
        x[: data.test_start], (data.y - data.persistence)[: data.test_start]
    )
    return FittedForecaster(model, x.shape[1], params, candidates)


def run_classical(path=None):
    data = prepare(load_data(path))
    scale = float(np.mean(abs(np.diff(data.y[: data.initial_train]))))
    test = slice(data.test_start, None)
    validation = slice(data.initial_train, data.test_start)
    rows = {}
    for name, prediction in [
        ("persistence", data.persistence),
        ("daily", data.daily),
        ("weekly", data.weekly),
    ]:
        rows[name] = {
            "validation_RMSE_MW": metrics(
                data.y[validation], prediction[validation], scale
            )["RMSE_MW"],
            "test": metrics(data.y[test], prediction[test], scale),
            "trained_parameters": 0,
        }
    for name, (kind, grid) in CLASSICAL.items():
        rows[name], _ = fit_evaluate(kind, grid, data.raw, data)
    for variant in ("reservoir", "hybrid"):
        predictions = []
        for seed in SEEDS:
            h = echo_features(data.windows, seed)
            x = h if variant == "reservoir" else np.column_stack([data.raw, h])
            rows[f"echo_{variant}_{seed}"], p = fit_evaluate(
                "ridge", RIDGE_GRID, x, data
            )
            predictions.append(p)
        rows[f"echo_{variant}_ensemble"] = {
            "test": metrics(data.y[test], np.mean(predictions, axis=0), scale),
            "rule": "Equal weights fixed before test; all three source seeds retained",
        }
    return {
        "task": "Classical-only rolling 30-minute hindcast",
        "splits": split_description(data),
        "models": rows,
    }
