import unittest
import numpy as np
from classical_studies.nonconvex import (
    objective,
    gradient,
    hessian,
    global_minima,
    optimize,
    run_study,
)


class NonconvexTests(unittest.TestCase):
    def test_every_declared_benchmark_run_meets_scientific_acceptance(self):
        rows = run_study()["records"]
        self.assertEqual(len(rows), 27)
        self.assertTrue(all(r["scientific_target_met"] for r in rows))

    def test_analytic_derivatives_match_independent_central_differences(self):
        for name in ("rosenbrock", "himmelblau"):
            for x in (
                np.array([-1.2, 1.0]),
                np.array([0.4, -0.7]),
                np.array([3.0, 2.0]),
            ):
                eps = 1e-5
                basis = np.eye(2) * eps
                g = np.array(
                    [
                        (objective(name, x + d) - objective(name, x - d)) / (2 * eps)
                        for d in basis
                    ]
                )
                h = np.column_stack(
                    [
                        (gradient(name, x + d) - gradient(name, x - d)) / (2 * eps)
                        for d in basis
                    ]
                )
                np.testing.assert_allclose(gradient(name, x), g, rtol=1e-7, atol=2e-6)
                np.testing.assert_allclose(hessian(name, x), h, rtol=1e-7, atol=2e-6)

    def test_minima_are_certified_zeros_with_positive_curvature(self):
        for name, count in [("rosenbrock", 1), ("himmelblau", 4)]:
            roots = global_minima(name)
            self.assertEqual(len(roots), count)
            for x in roots:
                self.assertLess(objective(name, x), 1e-20)
                self.assertLess(np.linalg.norm(gradient(name, x)), 1e-9)
                self.assertGreater(np.min(np.linalg.eigvalsh(hessian(name, x))), 0)

    def test_bfgs_reaches_references_from_difficult_starts(self):
        for name, starts in [
            ("rosenbrock", [[-1.5, -1.5], [-1.2, 1], [2, 2]]),
            ("himmelblau", [[-5, -5], [-5, 5], [5, -5], [5, 5]]),
        ]:
            for start in starts:
                custom = optimize(name, start, "bfgs")
                reference = optimize(name, start, "scipy_bfgs")
                self.assertTrue(custom["scientific_target_met"], custom)
                self.assertTrue(reference["scientific_target_met"], reference)
                self.assertLess(custom["objective"], 1e-10)

    def test_budget_exhaustion_is_not_silently_success(self):
        r = optimize(
            "rosenbrock", [-1.2, 1], max_iterations=1, method="gradient_descent"
        )
        self.assertFalse(r["scientific_target_met"])
        self.assertFalse(r["solver_success"])
        self.assertEqual(r["termination"], "iteration_budget")
        self.assertGreater(r["evaluations"]["function"], 1)

    def test_gradient_descent_monotonic_and_bad_inputs_rejected(self):
        r = optimize("himmelblau", [-5, 5], "gradient_descent", max_iterations=500)
        f = [v["objective"] for v in r["trace"]]
        self.assertTrue(all(a + 1e-12 >= b for a, b in zip(f, f[1:])))
        for x in ([1], [float("nan"), 1], [[1], [2]], [1j, 2], ["1", "2"]):
            with self.assertRaises(ValueError):
                optimize("rosenbrock", x)


if __name__ == "__main__":
    unittest.main()
