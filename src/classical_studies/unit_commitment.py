"""Audited MOPSO/QPSO on the original synthetic two-unit, four-hour case."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from time import perf_counter
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

P_MIN = np.array([20.0, 0.0])
P_MAX = np.array([100.0, 80.0])
DEMAND = np.array([60.0, 90.0, 110.0, 70.0])
COST = np.array([20.0, 40.0])
START_COST = np.array([100.0, 10.0])
EMISSIONS = np.array([0.9, 0.1])
START_EMISSIONS = np.array([2.0, 0.5])
LOW = np.array([6910.0, 75.5])
HIGH = np.array([12310.0, 291.5])


@dataclass(frozen=True)
class Schedule:
    generation: np.ndarray
    commitment: np.ndarray
    startup: np.ndarray


def startup_flags(commitment):
    u = np.asarray(commitment)
    if u.shape != (2, 4) or not np.isin(u, [0, 1]).all():
        raise ValueError("Invalid commitment matrix")
    previous = np.column_stack([np.zeros(2, dtype=int), u[:, :-1]])
    return ((u == 1) & (previous == 0)).astype(int)


def audit(schedule):
    p = np.asarray(schedule.generation)
    u = np.asarray(schedule.commitment)
    y = np.asarray(schedule.startup)
    if any(v.shape != (2, 4) for v in (p, u, y)):
        return False
    if any(v.dtype.kind not in "biuf" or np.iscomplexobj(v) for v in (p, u, y)):
        return False
    if not all(np.isfinite(v).all() for v in (p, u, y)):
        return False
    if not np.isin(u, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        return False
    return bool(
        np.all(p >= P_MIN[:, None] * u - 1e-8)
        and np.all(p <= P_MAX[:, None] * u + 1e-8)
        and np.allclose(p.sum(axis=0), DEMAND, rtol=0, atol=1e-8)
        and np.array_equal(y, startup_flags(u))
    )


def objectives(schedule):
    if not audit(schedule):
        raise ValueError("Cannot score an infeasible schedule")
    return np.array(
        [
            np.sum(
                COST[:, None] * schedule.generation
                + START_COST[:, None] * schedule.startup
            ),
            np.sum(
                EMISSIONS[:, None] * schedule.generation
                + START_EMISSIONS[:, None] * schedule.startup
            ),
        ]
    )


def project(raw):
    """Nearest per-hour feasible dispatch; no hidden cost/emissions repair weight."""
    if np.asarray(raw).dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError("Expected real numeric dispatch coordinates")
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (2, 4) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite 2-by-4 position")
    p = np.zeros_like(raw)
    u = np.zeros_like(raw, dtype=int)
    for hour, demand in enumerate(DEMAND):
        choices = []
        for mask in product((0, 1), repeat=2):
            on = np.array(mask)
            lo = P_MIN * on
            hi = P_MAX * on
            left = max(lo[0], demand - hi[1])
            right = min(hi[0], demand - lo[1])
            if left > right:
                continue
            first = np.clip(0.5 * (raw[0, hour] + demand - raw[1, hour]), left, right)
            generation = np.array([first, demand - first])
            choices.append(
                (
                    float(np.sum((generation - raw[:, hour]) ** 2)),
                    sum(mask),
                    mask,
                    generation,
                )
            )
        if not choices:
            raise ValueError("Demand is infeasible")
        _, _, mask, generation = min(choices, key=lambda r: r[:3])
        p[:, hour] = generation
        u[:, hour] = mask
    result = Schedule(p, u, startup_flags(u))
    if not audit(result):
        raise AssertionError("Projection failed independent feasibility audit")
    return result


def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def add_archive(archive, schedule, capacity=64):
    value = objectives(schedule)
    if any(
        dominates(v, value) or np.allclose(v, value, rtol=0, atol=1e-8)
        for _, v in archive
    ):
        return archive
    result = [(s, v) for s, v in archive if not dominates(value, v)] + [
        (schedule, value)
    ]
    if len(result) > capacity:
        scores = np.array([v for _, v in result])
        distance = np.zeros(len(result))
        for dim in range(2):
            order = np.argsort(scores[:, dim])
            span = np.ptp(scores[:, dim])
            distance[order[[0, -1]]] = np.inf
            if span > 0:
                distance[order[1:-1]] += (
                    scores[order[2:], dim] - scores[order[:-2], dim]
                ) / span
        result.pop(int(np.argmin(distance)))
    return result


def exact_reference(weight=1.0, emissions_cap=None):
    """Independent HiGHS mixed-integer formulation with exact startup constraints."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Weight must be in [0,1]")
    if emissions_cap is not None and not np.isfinite(emissions_cap):
        raise ValueError("Invalid emissions cap")
    # Variables: generation[8], on[8], start[8]. Units start off.
    pc = np.repeat(COST, 4)
    ec = np.repeat(EMISSIONS, 4)
    sy = np.repeat(START_COST, 4)
    ey = np.repeat(START_EMISSIONS, 4)
    c = np.r_[
        weight * pc / 10000 + (1 - weight) * ec / 300,
        np.zeros(8),
        weight * sy / 10000 + (1 - weight) * ey / 300,
    ]
    rows = []
    lower = []
    upper = []

    def constraint(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for hour in range(4):
        row = np.zeros(24)
        row[[hour, 4 + hour]] = 1
        constraint(row, DEMAND[hour], DEMAND[hour])
    for unit in range(2):
        for hour in range(4):
            k = 4 * unit + hour
            p = k
            u = 8 + k
            y = 16 + k
            row = np.zeros(24)
            row[p] = 1
            row[u] = -P_MIN[unit]
            constraint(row, 0, np.inf)
            row = np.zeros(24)
            row[p] = 1
            row[u] = -P_MAX[unit]
            constraint(row, -np.inf, 0)
            row = np.zeros(24)
            row[y] = 1
            row[u] = -1
            if hour > 0:
                row[u - 1] = 1
            constraint(row, 0, np.inf)
            row = np.zeros(24)
            row[y] = 1
            row[u] = -1
            constraint(row, -np.inf, 0)
            row = np.zeros(24)
            row[y] = 1
            if hour > 0:
                row[u - 1] = 1
            constraint(row, -np.inf, 1)
    if emissions_cap is not None:
        constraint(np.r_[ec, np.zeros(8), ey], -np.inf, emissions_cap)
    fit = milp(
        c,
        integrality=np.r_[np.zeros(8), np.ones(16)],
        bounds=Bounds(np.zeros(24), np.r_[np.repeat(P_MAX, 4), np.ones(16)]),
        constraints=LinearConstraint(np.array(rows), lower, upper),
        options={"mip_rel_gap": 1e-10, "time_limit": 30},
    )
    if not fit.success:
        raise RuntimeError("Exact reference failed: " + fit.message)
    schedule = Schedule(
        fit.x[:8].reshape(2, 4),
        np.rint(fit.x[8:16]).astype(int).reshape(2, 4),
        np.rint(fit.x[16:]).astype(int).reshape(2, 4),
    )
    return schedule, {
        "objectives": objectives(schedule).tolist(),
        "mip_gap": float(fit.mip_gap),
    }


def normalized_hypervolume(values):
    points = (np.asarray(values, dtype=float) - LOW) / (HIGH - LOW)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Invalid objective points")
    area = 0.0
    previous = 1.1
    for x, y in points[np.argsort(points[:, 0])]:
        if x < 1.1 and y < previous:
            area += (1.1 - x) * (previous - y)
            previous = y
    return float(area)


def run_swarm(method, *, seed=0, particles=30, iterations=150, capacity=64):
    if method not in ("mopso", "qpso"):
        raise ValueError("Unknown swarm method")
    for value, minimum in ((seed, 0), (particles, 2), (iterations, 1), (capacity, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("Invalid swarm configuration")
    if particles * iterations > 1_000_000 or capacity > 512:
        raise ValueError("Swarm exceeds the experiment budget")
    rng = np.random.default_rng(seed)
    start = perf_counter()
    raw = rng.uniform(0, P_MAX[None, :, None], size=(particles, 2, 4)) * (
        rng.random((particles, 2, 4)) < 0.5
    )
    positions = [project(v) for v in raw]
    personal = positions.copy()
    pvalues = [objectives(v) for v in personal]
    velocities = rng.uniform(-0.5, 0.5, size=(particles, 2, 4)) * P_MAX[None, :, None]
    archive = []
    evaluations = 0
    for s in positions:
        archive = add_archive(archive, s, capacity)
        evaluations += 1
    trace = []
    for iteration in range(iterations):
        mbest = np.mean([s.generation for s in personal], axis=0)
        beta = 1.0 - 0.5 * iteration / max(iterations - 1, 1)
        for i in range(particles):
            leader = archive[int(rng.integers(len(archive)))][0].generation
            x = positions[i].generation
            best = personal[i].generation
            if method == "mopso":
                v = (
                    0.7 * velocities[i]
                    + 1.4 * rng.random((2, 4)) * (best - x)
                    + 1.4 * rng.random((2, 4)) * (leader - x)
                )
                velocities[i] = np.clip(v, -0.5 * P_MAX[:, None], 0.5 * P_MAX[:, None])
                proposal = x + velocities[i]
            else:
                phi = rng.random((2, 4))
                attractor = phi * best + (1 - phi) * leader
                uniform = np.clip(
                    rng.random((2, 4)), np.finfo(float).eps, 1 - np.finfo(float).eps
                )
                sign = np.where(rng.random((2, 4)) < 0.5, -1.0, 1.0)
                proposal = attractor + sign * beta * abs(mbest - x) * np.log(
                    1 / uniform
                )
            positions[i] = project(np.clip(proposal, 0, P_MAX[:, None]))
            value = objectives(positions[i])
            evaluations += 1
            if dominates(value, pvalues[i]) or (
                not dominates(pvalues[i], value) and rng.random() < 0.5
            ):
                personal[i] = positions[i]
                pvalues[i] = value
            archive = add_archive(archive, positions[i], capacity)
        if iteration % 10 == 0 or iteration == iterations - 1:
            trace.append(
                {
                    "iteration": iteration + 1,
                    "hypervolume": normalized_hypervolume([v for _, v in archive]),
                }
            )
    records = [
        {
            "cost": float(v[0]),
            "emissions": float(v[1]),
            "generation": s.generation.tolist(),
            "commitment": s.commitment.tolist(),
            "startup": s.startup.tolist(),
        }
        for s, v in sorted(archive, key=lambda r: r[1][0])
    ]
    hv = normalized_hypervolume([v for _, v in archive])
    return {
        "method": method,
        "seed": seed,
        "particles": particles,
        "iterations": iterations,
        "candidate_evaluations": evaluations,
        "budget_definition": "Initial particles plus proposed candidates; repeated internal scoring/audits are not additional candidates.",
        "archive_capacity": capacity,
        "archive_size": len(archive),
        "all_schedules_feasible": all(audit(s) for s, _ in archive),
        "normalized_hypervolume": hv,
        "continuous_reference_hypervolume": 0.71,
        "reference_hypervolume_fraction": hv / 0.71,
        "seconds": perf_counter() - start,
        "frontier": records,
        "trace": trace,
    }


def run_study():
    refs = []
    for cap in np.linspace(LOW[1], HIGH[1], 17):
        _, record = exact_reference(1.0, float(cap))
        record["emissions_cap"] = float(cap)
        refs.append(record)
        cost, emissions = record["objectives"]
        if (
            abs(
                (cost - LOW[0]) / (HIGH[0] - LOW[0])
                + (emissions - LOW[1]) / (HIGH[1] - LOW[1])
                - 1
            )
            > 1e-7
        ):
            raise AssertionError(
                "MILP does not confirm the derived toy-case Pareto line"
            )
    return {
        "scope": "Synthetic two-unit, four-hour commitment/dispatch; exact balance, minimum/maximum output, initially-off startup logic. Zero-output commitment is allowed for the zero-minimum unit.",
        "limits": "No ramps, minimum up/down times, network, reserves or real grid data. QPSO is a classical randomized optimizer, without qubits or quantum execution.",
        "reference": "Both units must start at least once. Total A generation lies in [50,320] MWh; one-start feasible commitment windows cover this interval. Thus C=13310-20*A and E=35.5+.8*A give the complete continuous Pareto line.",
        "normalization": {
            "ideal": LOW.tolist(),
            "nadir": HIGH.tolist(),
            "hypervolume_reference_point": [1.1, 1.1],
        },
        "exact_references": refs,
        "records": [
            run_swarm(method, seed=seed)
            for seed in (0, 1, 7, 19, 41)
            for method in ("mopso", "qpso")
        ],
    }
