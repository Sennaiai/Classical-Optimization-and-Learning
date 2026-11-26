import unittest
from itertools import product
import numpy as np
from classical_studies.fuzzy import (
    alpha_cut,
    solve,
    vertices,
    REQUIREMENTS,
    PROFITS,
    RESOURCES,
)


class FuzzyTests(unittest.TestCase):
    def test_cut_endpoints_and_nesting(self):
        np.testing.assert_allclose(alpha_cut([2, 3, 5], 0), [2, 5])
        np.testing.assert_allclose(alpha_cut([2, 3, 5], 1), [3, 3])
        lo, hi = alpha_cut([2, 3, 5], 0.5)
        self.assertLess(lo, hi)
        for invalid in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                alpha_cut([2, 3, 5], invalid)
        with self.assertRaises(ValueError):
            alpha_cut([3, 2, 5], 0.5)

    def test_all_policies_match_vertex_reference(self):
        for policy in ("legacy", "robust", "nominal", "optimistic"):
            for alpha in np.linspace(0, 1, 9):
                r = solve(alpha, policy)
                self.assertLess(r["reference_absolute_error"], 1e-8)
                self.assertGreaterEqual(min(r["resource_slack"]), -1e-8)

    def test_nominal_invariance_and_conservative_envelope(self):
        nominal = solve(0, "nominal")["profit"]
        for a in (0, 0.25, 0.75, 1):
            self.assertAlmostEqual(solve(a, "nominal")["profit"], nominal)
            self.assertLessEqual(
                solve(a, "robust")["profit"], solve(a, "legacy")["profit"] + 1e-9
            )
            self.assertLessEqual(
                solve(a, "legacy")["profit"], solve(a, "optimistic")["profit"] + 1e-9
            )
        self.assertAlmostEqual(solve(1, "robust")["profit"], nominal)

    def test_robust_solution_survives_all_cut_box_corners(self):
        for a in (0, 0.5, 1):
            r = solve(a, "robust")
            x = np.array(r["production"])
            self.assertTrue(
                np.all(REQUIREMENTS @ x <= np.array(r["resource_limits"]) + 1e-9)
            )
            self.assertAlmostEqual(r["profit"], np.dot(r["profits_per_unit"], x))
            plo, phi = alpha_cut(PROFITS, a)
            rlo, rhi = alpha_cut(RESOURCES, a)
            for rc in product(*zip(rlo, rhi)):
                self.assertTrue(np.all(REQUIREMENTS @ x <= np.array(rc) + 1e-9))
            for pc in product(*zip(plo, phi)):
                self.assertGreaterEqual(float(np.dot(pc, x)), r["profit"] - 1e-9)
        self.assertEqual(len(vertices([[1, 0], [0, 1]], [1, 1])), 4)


if __name__ == "__main__":
    unittest.main()
