import unittest
import numpy as np
from classical_studies.unit_commitment import (
    Schedule,
    audit,
    objectives,
    project,
    startup_flags,
    exact_reference,
    LOW,
    HIGH,
    normalized_hypervolume,
    run_swarm,
    add_archive,
)


class CommitmentTests(unittest.TestCase):
    def test_hypervolume_matches_hand_integrated_rectangles(self):
        points = [
            LOW + (HIGH - LOW) * p for p in np.array([[0, 1], [0.5, 0.5], [1, 0]])
        ]
        self.assertAlmostEqual(normalized_hypervolume(points), 0.46, places=12)
        self.assertAlmostEqual(
            normalized_hypervolume(points[::-1] + [points[1]]), 0.46, places=12
        )

    def test_projection_rejects_the_original_subminimum_and_startup_bugs(self):
        s = project(np.array([[1.0, 2.0, 3.0, 4.0], [59.0, 88.0, 107.0, 66.0]]))
        self.assertTrue(audit(s))
        self.assertTrue(np.all((s.generation[0] == 0) | (s.generation[0] >= 20)))
        wrong = Schedule(s.generation, s.commitment, np.zeros((2, 4), int))
        self.assertFalse(audit(wrong))
        with self.assertRaises(ValueError):
            objectives(wrong)
        with self.assertRaises(ValueError):
            project(np.full((2, 4), np.nan))

    def test_projection_is_feasible_for_extreme_and_random_positions(self):
        rng = np.random.default_rng(197)
        for raw in [np.zeros((2, 4)), np.ones((2, 4)) * 1e5] + list(
            rng.normal(50, 100, size=(100, 2, 4))
        ):
            self.assertTrue(audit(project(raw)))

    def test_exact_milp_matches_analytic_endpoints_and_interior(self):
        for weight, expected in ((1.0, [LOW[0], HIGH[1]]), (0.0, [HIGH[0], LOW[1]])):
            schedule, record = exact_reference(weight)
            self.assertTrue(audit(schedule))
            np.testing.assert_allclose(
                record["objectives"], expected, rtol=0, atol=1e-6
            )
            self.assertLessEqual(record["mip_gap"], 1e-9)
        for cap in np.linspace(LOW[1], HIGH[1], 7):
            _, r = exact_reference(1.0, cap)
            c, e = r["objectives"]
            self.assertAlmostEqual(
                (c - LOW[0]) / (HIGH[0] - LOW[0]) + (e - LOW[1]) / (HIGH[1] - LOW[1]),
                1.0,
                places=8,
            )

    def test_swarm_comparison_uses_equal_budgets_and_valid_archives(self):
        for method in ("mopso", "qpso"):
            a = run_swarm(method, seed=7, particles=6, iterations=8, capacity=16)
            b = run_swarm(method, seed=7, particles=6, iterations=8, capacity=16)
            self.assertEqual(a["candidate_evaluations"], 54)
            self.assertEqual(a["frontier"], b["frontier"])
            self.assertTrue(a["all_schedules_feasible"])
            self.assertLessEqual(a["reference_hypervolume_fraction"], 1 + 1e-9)
            values = []
            for row in a["frontier"]:
                s = Schedule(
                    np.array(row["generation"]),
                    np.array(row["commitment"]),
                    np.array(row["startup"]),
                )
                self.assertTrue(audit(s))
                np.testing.assert_allclose(
                    objectives(s), [row["cost"], row["emissions"]]
                )
                values.append([row["cost"], row["emissions"]])
            self.assertEqual(
                normalized_hypervolume(values), a["normalized_hypervolume"]
            )

    def test_archive_deduplicates_and_excludes_infeasible_records(self):
        s = project(np.zeros((2, 4)))
        a = add_archive([], s)
        self.assertEqual(len(add_archive(a, s)), 1)
        with self.assertRaises(ValueError):
            add_archive(
                [], Schedule(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)))
            )


if __name__ == "__main__":
    unittest.main()
