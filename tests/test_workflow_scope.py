import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scope", ROOT / ".github/scripts/select_studies.py"
)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


class ScopeTests(unittest.TestCase):
    def test_documentation_does_not_rerun_studies(self):
        self.assertEqual(
            scope.select(["README.md", "projects/optimization/nonconvex/README.md"]), []
        )

    def test_each_numerical_module_selects_only_its_study(self):
        for name in scope.CATALOG:
            self.assertEqual(scope.select([f"src/classical_studies/{name}.py"]), [name])
            self.assertEqual(scope.select([scope.CATALOG[name]["notebook"]]), [name])

    def test_classical_library_does_not_rerun_notebook_studies(self):
        for path in (
            "src/classical_studies/baselines/forecasting/models.py",
            "tests/allocation/test_solver.py",
            "tests/test_molecular_baseline.py",
        ):
            self.assertEqual(scope.select([path]), ["baselines"])

    def test_environment_and_workflow_changes_validate_every_group(self):
        self.assertEqual(scope.select(["pyproject.toml"]), list(scope.ALL))
        self.assertEqual(
            scope.select([".github/workflows/quality-control.yml"]), list(scope.ALL)
        )


if __name__ == "__main__":
    unittest.main()
