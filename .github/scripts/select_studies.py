"""Select only studies affected by changed source; documentation is lightweight."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "study_catalog.json").read_text())
ALL = tuple(CATALOG) + ("baselines",)


def select(paths):
    selected = set()
    for path in paths:
        if path in (
            "pyproject.toml",
            "environment.yml",
            "requirements-dev.txt",
            "study_catalog.json",
        ) or path.startswith((".github/scripts/", ".github/workflows/")):
            selected.update(ALL)
        elif (
            path.startswith("src/classical_studies/baselines/")
            or path.startswith("tests/allocation/")
            or path.startswith("tests/test_baseline")
            or path
            in (
                "tests/test_forecasting_baselines.py",
                "tests/test_molecular_baseline.py",
            )
        ):
            selected.add("baselines")
        elif path in (
            "src/classical_studies/run.py",
            "src/classical_studies/__init__.py",
        ):
            selected.update(CATALOG)
        else:
            for name, spec in CATALOG.items():
                project = str(Path(spec["notebook"]).parent) + "/"
                if (
                    path == f"src/classical_studies/{name}.py"
                    or path == f"tests/test_{name}.py"
                    or (path.startswith(project) and not path.endswith(".md"))
                ):
                    selected.add(name)
    return [name for name in ALL if name in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    if args.manual:
        names = list(ALL)
    else:
        base = os.environ.get("BASE_SHA", "")
        head = os.environ.get("HEAD_SHA", "")
        if not re.fullmatch("[0-9a-f]{40}", head):
            raise ValueError("Invalid head SHA")
        if base == "0" * 40:
            names = list(ALL)
        else:
            if not re.fullmatch("[0-9a-f]{40}", base):
                raise ValueError("Invalid base SHA")
            paths = subprocess.check_output(
                ["git", "diff", "--name-only", base, head], cwd=ROOT, text=True
            ).splitlines()
            names = select(paths)
    result = {"include": [{"study": n} for n in names]}
    payload = json.dumps(result)
    print(payload)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write(
                "matrix=" + payload + "\nhas_work=" + str(bool(names)).lower() + "\n"
            )


if __name__ == "__main__":
    main()
