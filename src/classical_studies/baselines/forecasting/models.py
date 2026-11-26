"""Classical-only extraction of the validated grid-forecasting controls."""

from time import perf_counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data_contract import SEEDS, load_data, prepare, split_description


def metrics(y, pred, scale):
    y, pred = np.asarray(y), np.asarray(pred)
    if (
        y.shape != pred.shape
        or y.ndim != 1
        or y.size == 0
        or not np.isfinite(y).all()
        or not np.isfinite(pred).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid forecast metrics input")
    return {
        "MAE_MW": float(np.mean(abs(y - pred))),
        "RMSE_MW": float(np.sqrt(np.mean((y - pred) ** 2))),
        "MASE": float(np.mean(abs(y - pred)) / scale),
    }


def model_factory(kind, params):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=100, random_state=19, n_jobs=1, **params
        )
    if kind == "boost":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            early_stopping=False,
            random_state=19,
            **params,
        )
    raise ValueError("Unknown model family")


def select_model(kind, grid, x, data):
    """This function cannot inspect heldout labels: every candidate uses folds()."""
    if x.shape[0] != len(data.y) or not np.isfinite(x).all():
        raise ValueError("Misaligned/non-finite feature matrix")
    residual = data.y - data.persistence
    scale = float(np.mean(abs(np.diff(data.y[: data.initial_train]))))
    candidates = []
    for params in grid:
        truth, pred = [], []
        for train, validation in data.folds():
            if train[-1] >= validation[0] or validation[-1] >= data.test_start:
                raise ValueError("Validation must be chronological and pre-test")
            model = model_factory(kind, params).fit(x[train], residual[train])
            truth.extend(data.y[validation])
            pred.extend(data.persistence[validation] + model.predict(x[validation]))
        candidates.append(
            {"params": params, **metrics(np.array(truth), np.array(pred), scale)}
        )
    best = min(candidates, key=lambda p: p["RMSE_MW"])
    return dict(best["params"]), candidates


def fit_evaluate(kind, grid, x, data):
    params, candidates = select_model(kind, grid, x, data)
    start = perf_counter()
    model = model_factory(kind, params).fit(
        x[: data.test_start], (data.y - data.persistence)[: data.test_start]
    )
    seconds = perf_counter() - start
    p = data.persistence + model.predict(x)
    scale = float(np.mean(abs(np.diff(data.y[: data.initial_train]))))
    row = {
        "selected_params": params,
        "validation_candidates": candidates,
        "validation_RMSE_MW": min(c["RMSE_MW"] for c in candidates),
        "train": metrics(data.y[: data.test_start], p[: data.test_start], scale),
        "test": metrics(data.y[data.test_start :], p[data.test_start :], scale),
        "features": x.shape[1],
        "fit_seconds": seconds,
        "test_to_train_MAE_ratio": float(
            np.mean(abs(data.y[data.test_start :] - p[data.test_start :]))
            / max(np.mean(abs(data.y[: data.test_start] - p[: data.test_start])), 1e-12)
        ),
    }
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final, "coef_"):
        row["trained_readout_coefficients_including_intercept"] = int(
            final.coef_.size + 1
        )
    if hasattr(final, "estimators_"):
        row["tree_nodes"] = int(sum(t.tree_.node_count for t in final.estimators_))
    return row, p[data.test_start :]


def paired_day_bootstrap(y, pred, reference, repeats=1000):
    """Paired 48-step nonoverlapping block means; conditional exploratory interval.

    Fixed 24-hour elapsed blocks, not calendar-day/DST bins. Incomplete final
    block excluded. Does not account for model selection or seasonal dependence.
    """
    differences = abs(y - pred) - abs(y - reference)
    full = len(differences) // 48
    if full < 5:
        raise ValueError("Too few test blocks for temporal bootstrap")
    blocks = differences[: full * 48].reshape(full, 48).mean(1)
    rng = np.random.default_rng(812)
    means = blocks[rng.integers(full, size=(repeats, full))].mean(1)
    return {
        "MAE_difference_MW": float(differences.mean()),
        "block_mean_difference_MW": float(blocks.mean()),
        "conditional_95pct_interval_MW": np.quantile(means, [0.025, 0.975]).tolist(),
        "blocks": full,
        "block_steps": 48,
        "resamples": repeats,
        "interpretation": "Exploratory conditional interval; no multiplicity/selection correction or advantage claim",
    }


RIDGE_GRID = [{"alpha": a} for a in (0.1, 10.0, 1000.0)]

CLASSICAL = {
    "raw_ridge": ("ridge", RIDGE_GRID),
    "random_forest": (
        "rf",
        [
            {"max_depth": 12, "min_samples_leaf": 5},
            {"max_depth": None, "min_samples_leaf": 12},
        ],
    ),
    "gradient_boosting": (
        "boost",
        [
            {"max_leaf_nodes": 15, "l2_regularization": 1.0},
            {"max_leaf_nodes": 31, "l2_regularization": 10.0},
        ],
    ),
}
