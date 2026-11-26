"""Run one bounded study and save numerical evidence with source provenance."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

STUDIES = ("nonconvex", "unit_commitment", "fuzzy", "option_pricing")


def run(study, output):
    if study not in STUDIES:
        raise ValueError("Unknown study")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "classical-studies-mpl")
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    module = importlib.import_module("classical_studies." + study)
    start = perf_counter()
    report = module.run_study()
    report["total_study_seconds"] = perf_counter() - start
    report["provenance"] = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib")
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(module.__file__), Path(__file__))
        },
        "thread_settings": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot(study, report, output / "benchmark.svg")
    print(
        json.dumps(
            {
                "study": study,
                "records": len(report["records"]),
                "seconds": report["total_study_seconds"],
            }
        ),
        flush=True,
    )
    return report


def plot(study, report, path):
    import numpy as np
    import matplotlib.pyplot as plt

    rows = report["records"]
    if study == "fuzzy":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for policy in ("legacy", "robust", "nominal", "optimistic"):
            r = [v for v in rows if v["policy"] == policy]
            axes[0].plot(
                [v["alpha"] for v in r],
                [v["profit"] for v in r],
                marker="o",
                label=policy,
            )
        axes[0].set(
            xlabel="Membership level α", ylabel="Objective profit (synthetic currency)"
        )
        r = [v for v in rows if v["policy"] == "robust"]
        for index in (0, 1):
            axes[1].plot(
                [v["alpha"] for v in r],
                [v["production"][index] for v in r],
                marker="o",
                label=f"Product {index + 1}",
            )
        axes[1].set(xlabel="Membership level α", ylabel="Robust production (units)")
    elif study == "option_pricing":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, case in zip(axes, ("out_of_money", "at_money", "in_money")):
            for method in ("mc", "antithetic", "scrambled_sobol", "scrambled_halton"):
                group = [v for v in rows if v["case"] == case and v["method"] == method]
                budgets = sorted({v["total_payoff_evaluations"] for v in group})
                errors = [
                    np.sqrt(
                        np.mean(
                            [
                                v["absolute_error"] ** 2
                                for v in group
                                if v["total_payoff_evaluations"] == b
                            ]
                        )
                    )
                    for b in budgets
                ]
                ax.loglog(budgets, errors, marker="o", label=method)
            ax.set(
                title=case.replace("_", " "),
                xlabel="Total payoff evaluations",
                ylabel="RMSE across four seeds (currency)",
            )
    elif study == "nonconvex":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, name in zip(axes, ("rosenbrock", "himmelblau")):
            start = next(v["start"] for v in rows if v["objective_name"] == name)
            for method in ("gradient_descent", "bfgs", "scipy_bfgs"):
                r = next(
                    v
                    for v in rows
                    if v["objective_name"] == name
                    and v["start"] == start
                    and v["method"] == method
                )
                ax.semilogy(
                    [t["iteration"] for t in r["trace"]],
                    [max(t["objective"], 1e-25) for t in r["trace"]],
                    marker="." if method == "scipy_bfgs" else None,
                    label=method,
                )
            ax.set(
                title=f"{name}; start {start}",
                xlabel="Iterations",
                ylabel="Objective (diagnostic floor 1e−25)",
            )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ref = report["exact_references"]
        axes[0].plot(
            [v["objectives"][0] for v in ref],
            [v["objectives"][1] for v in ref],
            color="black",
            label="Exact continuous Pareto line",
        )
        for method in ("mopso", "qpso"):
            r = [v for v in rows if v["method"] == method]
            first = r[0]["frontier"]
            axes[0].scatter(
                [v["cost"] for v in first],
                [v["emissions"] for v in first],
                s=12,
                label=method + "; seed 0",
            )
            steps = [v["iteration"] for v in r[0]["trace"]]
            values = np.array([[t["hypervolume"] for t in v["trace"]] for v in r])
            axes[1].plot(steps, np.median(values, axis=0), label=method)
            axes[1].fill_between(
                steps, values.min(axis=0), values.max(axis=0), alpha=0.15
            )
        axes[0].set(
            xlabel="Cost (synthetic currency)", ylabel="Emissions (assumed tonnes CO₂)"
        )
        axes[1].axhline(
            0.71, color="black", ls="--", label="Continuous-front reference"
        )
        axes[1].set(xlabel="Iterations", ylabel="Normalized dominated hypervolume")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    path = Path(path)
    if path.suffix == ".svg":
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
