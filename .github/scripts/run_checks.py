"""Run one group's tests and controlled notebook execution."""

import argparse, json, os, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "study_catalog.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=[*CATALOG, "baselines"])
    p.add_argument("--no-notebook", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    patterns = (
        [
            "test_baseline*.py",
            "test_forecasting_baselines.py",
            "test_molecular_baseline.py",
        ]
        if a.study == "baselines"
        else [CATALOG[a.study]["tests"]]
    )
    for pattern in patterns:
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=pattern))
    if a.study == "baselines":
        suite.addTests(loader.discover(str(ROOT / "tests/allocation")))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    artifact = ROOT / "artifacts" / a.study
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "validation.json").write_text(
        json.dumps(
            {
                "study": a.study,
                "tests": result.testsRun,
                "failures": 0,
                "scientific_scope": "Synthetic/component validation; predictive performance is not inferred from test counts.",
            },
            indent=2,
        )
        + "\n"
    )
    if a.study != "baselines" and not a.no_notebook:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / ".github/scripts/execute_notebooks.py"),
                "--study",
                a.study,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
