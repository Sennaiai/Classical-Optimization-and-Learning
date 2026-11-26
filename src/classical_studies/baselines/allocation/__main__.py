import argparse, json
from . import (
    solve_decomposed,
    score,
    refine_allocation,
    load_public_catalog,
    load_experiment_catalogs,
)
from .fixtures import small_cases

p = argparse.ArgumentParser()
p.add_argument(
    "--catalog", choices=["fixture", "provided", "expanded"], default="fixture"
)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
if a.catalog == "fixture":
    instance = small_cases()[0]
elif a.catalog == "provided":
    instance = load_public_catalog()[0]
else:
    instance = load_experiment_catalogs()[1]
assignment, run = solve_decomposed(instance, seed=a.seed)
assignment, refinement = refine_allocation(instance, assignment)
print(
    json.dumps(
        {
            "catalog": a.catalog,
            "score": score(instance, assignment),
            "seed": a.seed,
            "solver_seconds": run["seconds"],
        },
        indent=2,
    )
)
