"""Two-dimensional nonconvex optimization with analytic and numerical references."""

from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize


def point(x):
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError("Expected real numeric coordinates")
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError("Expected two finite coordinates")
    return x


def objective(name, x):
    x, y = point(x)
    if name == "rosenbrock":
        return float((1 - x) ** 2 + 100 * (y - x * x) ** 2)
    if name == "himmelblau":
        return float((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2)
    raise ValueError("Unknown objective")


def gradient(name, x):
    x, y = point(x)
    if name == "rosenbrock":
        return np.array([2 * (x - 1) - 400 * x * (y - x * x), 200 * (y - x * x)])
    if name == "himmelblau":
        return np.array(
            [
                4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
                2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
            ]
        )
    raise ValueError("Unknown objective")


def hessian(name, x):
    x, y = point(x)
    if name == "rosenbrock":
        return np.array([[2 - 400 * y + 1200 * x * x, -400 * x], [-400 * x, 200]])
    if name == "himmelblau":
        return np.array(
            [
                [12 * x * x + 4 * y - 42, 4 * (x + y)],
                [4 * (x + y), 4 * x + 12 * y * y - 26],
            ]
        )
    raise ValueError("Unknown objective")


def global_minima(name):
    if name == "rosenbrock":
        return np.array([[1.0, 1.0]])
    if name == "himmelblau":
        # Zero residuals imply y=11-x² and x⁴-22x²+x+114=0.
        roots = np.roots([1.0, 0.0, -22.0, 1.0, 114.0])
        return np.array(
            sorted(
                (float(x.real), float(11 - x.real * x.real))
                for x in roots
                if abs(x.imag) < 1e-10
            )
        )
    raise ValueError("Unknown objective")


def optimize(name, start, method="bfgs", *, tolerance=1e-7, max_iterations=40000):
    x = point(start).copy()
    initial = x.copy()
    global_minima(name)
    if method not in ("gradient_descent", "bfgs", "scipy_bfgs"):
        raise ValueError("Unknown optimizer")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= 100000
    ):
        raise ValueError("Invalid iteration budget")
    if (
        isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
        or not 1e-12 <= tolerance <= 1e-2
    ):
        raise ValueError("Invalid gradient tolerance")
    counts = {"function": 0, "gradient": 0}
    trace = []
    begin = perf_counter()

    def fun(z):
        counts["function"] += 1
        return objective(name, z)

    def jac(z):
        counts["gradient"] += 1
        return gradient(name, z)

    def record(i, z, f, g):
        if i < 200 or i % 100 == 0:
            trace.append(
                {
                    "iteration": int(i),
                    "x": z.tolist(),
                    "objective": float(f),
                    "gradient_inf": float(np.linalg.norm(g, np.inf)),
                }
            )

    if method == "scipy_bfgs":
        fit = minimize(
            fun,
            x,
            jac=jac,
            method="BFGS",
            options={"gtol": tolerance, "maxiter": max_iterations, "norm": np.inf},
        )
        x = fit.x
        value = objective(name, x)
        g = gradient(name, x)
        iterations = int(fit.nit)
        status = str(fit.message)
        solver_success = bool(fit.success)
    else:
        value = fun(x)
        g = jac(x)
        inverse = np.eye(2)
        status = "iteration_budget"
        solver_success = False
        iterations = 0
        record(0, x, value, g)
        for iteration in range(max_iterations):
            if np.linalg.norm(g, np.inf) <= tolerance:
                solver_success = True
                status = "gradient_tolerance"
                break
            direction = -(inverse @ g) if method == "bfgs" else -g
            if not np.isfinite(direction).all() or np.dot(direction, g) >= 0:
                direction = -g
                inverse = np.eye(2)
            slope = float(g @ direction)
            alpha = 1.0
            accepted = False
            for _ in range(55):
                trial = x + alpha * direction
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_value = (
                        fun(trial) if np.isfinite(trial).all() else float("inf")
                    )
                if (
                    np.isfinite(trial_value)
                    and trial_value <= value + 1e-4 * alpha * slope
                ):
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                status = "line_search_failed"
                break
            new_gradient = jac(trial)
            s = trial - x
            y = new_gradient - g
            if method == "bfgs":
                curvature = float(s @ y)
                if (
                    curvature > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y)
                    and curvature > 0
                ):
                    rho = 1 / curvature
                    v = np.eye(2) - rho * np.outer(s, y)
                    inverse = v @ inverse @ v.T + rho * np.outer(s, s)
                else:
                    inverse = np.eye(2)
            x, value, g = trial, trial_value, new_gradient
            iterations = iteration + 1
            record(iterations, x, value, g)
        if np.linalg.norm(g, np.inf) <= tolerance:
            solver_success = True
            status = "gradient_tolerance"
    nearest = float(np.min(np.linalg.norm(global_minima(name) - x, axis=1)))
    gradient_norm = float(np.linalg.norm(g, np.inf))
    scientific_target = bool(
        np.isfinite(value)
        and value <= 1e-10
        and gradient_norm <= 1e-6
        and nearest <= 1e-5
    )
    final = {
        "iteration": iterations,
        "x": x.tolist(),
        "objective": float(value),
        "gradient_inf": gradient_norm,
    }
    if not trace or trace[-1] != final:
        trace.append(final)
    return {
        "objective_name": name,
        "method": method,
        "gradient_tolerance": tolerance,
        "max_iterations": max_iterations,
        "start": initial.tolist(),
        "solution": x.tolist(),
        "objective": float(value),
        "gradient_inf": gradient_norm,
        "nearest_global_minimum_distance": nearest,
        "solver_success": solver_success,
        "termination": status,
        "scientific_target_met": scientific_target,
        "iterations": iterations,
        "evaluations": counts,
        "seconds": perf_counter() - begin,
        "trace": trace,
    }


def run_study():
    starts = {
        "rosenbrock": [[-1.5, -1.5], [-1.2, 1.0], [0.0, 0.0], [2.0, 2.0]],
        "himmelblau": [[0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]],
    }
    return {
        "scientific_target": "objective <= 1e-10, gradient infinity norm <= 1e-6, nearest certified zero <= 1e-5",
        "records": [
            optimize(name, start, method)
            for name, points in starts.items()
            for start in points
            for method in ("gradient_descent", "bfgs", "scipy_bfgs")
        ],
    }
