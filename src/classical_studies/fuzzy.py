"""Alpha-cut resource allocation; membership levels are not probabilities."""

from __future__ import annotations
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import linprog

PROFITS = np.array([[4.5, 5.0, 5.5], [2.8, 3.0, 3.2]])
RESOURCES = np.array([[38.0, 40.0, 42.0], [28.0, 30.0, 32.0]])
REQUIREMENTS = np.array([[2.0, 1.0], [1.0, 1.5]])


def alpha_cut(triangles, alpha):
    t = np.asarray(triangles, dtype=float)
    if (
        t.ndim < 1
        or t.shape[-1] != 3
        or not np.isfinite(t).all()
        or np.any(t[..., 0] > t[..., 1])
        or np.any(t[..., 1] > t[..., 2])
    ):
        raise ValueError("Expected finite ordered triangular numbers")
    if (
        isinstance(alpha, bool)
        or not np.isscalar(alpha)
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be in [0, 1]")
    return t[..., 0] + alpha * (t[..., 1] - t[..., 0]), t[..., 2] - alpha * (
        t[..., 2] - t[..., 1]
    )


def vertices(A, b):
    """Independent two-dimensional polygon enumeration, including nonnegativity."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    if (
        A.ndim != 2
        or A.shape[1] != 2
        or b.shape != (len(A),)
        or not np.isfinite(A).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Invalid two-variable constraints")
    constraints = np.vstack([A, -np.eye(2)])
    limits = np.r_[b, 0.0, 0.0]
    result = []
    for i, j in combinations(range(len(limits)), 2):
        matrix = constraints[[i, j]]
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        point = np.linalg.solve(matrix, limits[[i, j]])
        if np.all(constraints @ point <= limits + 1e-9) and not any(
            np.allclose(point, p, atol=1e-10, rtol=0) for p in result
        ):
            result.append(point)
    return np.asarray(result).reshape(-1, 2)


def solve(alpha, policy="robust"):
    plo, phi = alpha_cut(PROFITS, alpha)
    rlo, rhi = alpha_cut(RESOURCES, alpha)
    # Legacy policy retained explicitly: midpoint profits, lower resource limits.
    policies = {
        "robust": (plo, rlo),
        "legacy": ((plo + phi) / 2, rlo),
        "nominal": (PROFITS[:, 1], RESOURCES[:, 1]),
        "optimistic": (phi, rhi),
    }
    if policy not in policies:
        raise ValueError("Unknown alpha-cut policy")
    profits, limits = policies[policy]
    start = perf_counter()
    fit = linprog(
        -profits, A_ub=REQUIREMENTS, b_ub=limits, bounds=[(0, None)] * 2, method="highs"
    )
    elapsed = perf_counter() - start
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"LP failed: {fit.message}")
    points = vertices(REQUIREMENTS, limits)
    reference = float(np.max(points @ profits))
    value = float(profits @ fit.x)
    slack = limits - REQUIREMENTS @ fit.x
    if abs(value - reference) > 1e-8 or np.min(slack) < -1e-8 or np.min(fit.x) < -1e-8:
        raise AssertionError("LP disagrees with independent feasible-vertex reference")
    return {
        "alpha": float(alpha),
        "policy": policy,
        "production": fit.x.tolist(),
        "profits_per_unit": profits.tolist(),
        "resource_limits": limits.tolist(),
        "profit": value,
        "reference_profit": reference,
        "resource_slack": slack.tolist(),
        "reference_absolute_error": abs(value - reference),
        "solve_seconds": elapsed,
    }


def run_study():
    rows = [
        solve(float(alpha), policy)
        for policy in ("legacy", "robust", "nominal", "optimistic")
        for alpha in np.linspace(0, 1, 9)
    ]
    return {
        "case": "Synthetic two-product resource allocation; no carbon-storage model",
        "interpretation": "Robust policy optimizes worst-case profit over each alpha-cut box; alpha is a membership level, not a confidence or probability.",
        "records": rows,
    }
