import unittest
import numpy as np
from classical_studies.option_pricing import (
    Contract,
    black_scholes,
    discounted_payoffs,
    estimate,
)


class PricingTests(unittest.TestCase):
    def test_reference_and_analytic_limits(self):
        self.assertAlmostEqual(black_scholes(), 10.450583572185565, places=11)
        self.assertEqual(black_scholes(Contract(maturity=0)), 0.0)
        self.assertAlmostEqual(black_scholes(Contract(strike=0)), 100.0)
        self.assertAlmostEqual(
            black_scholes(Contract(volatility=0)), 100 - 100 * np.exp(-0.05)
        )
        self.assertGreater(black_scholes(Contract(spot=120)), black_scholes())
        self.assertGreater(black_scholes(Contract(volatility=0.4)), black_scholes())

    def test_risk_neutral_discounting_and_negative_rate(self):
        for rate in (-0.03, 0, 0.05):
            c = Contract(rate=rate, volatility=0)
            self.assertTrue(
                np.allclose(discounted_payoffs(np.zeros(4), c), black_scholes(c))
            )

    def test_equal_payoff_budgets_reproducible_and_finite(self):
        for method in ("mc", "antithetic", "scrambled_sobol", "scrambled_halton"):
            a = estimate(method, 128, replications=8)
            b = estimate(method, 128, replications=8)
            self.assertEqual(a["price"], b["price"])
            self.assertEqual(a["total_payoff_evaluations"], 1024)
            self.assertEqual(len(a["replication_prices"]), 8)
            self.assertTrue(np.isfinite(a["approximate_95_percent_interval"]).all())
            self.assertGreater(a["replicate_standard_error"], 0)

    def test_large_budget_matches_independent_reference(self):
        result = estimate("scrambled_sobol", 8192, replications=16)
        self.assertLess(result["absolute_error"], 0.02)
        # Reference error rather than requiring every random 95% interval to cover.
        self.assertLess(result["replicate_standard_error"], 0.02)

    def test_bad_parameters_and_budgets_rejected(self):
        for c in (
            Contract(spot=0),
            Contract(volatility=-1),
            Contract(rate=float("nan")),
            Contract(maturity=True),
        ):
            with self.assertRaises(ValueError):
                black_scholes(c)
        for kwargs in (
            {"n": 127},
            {"n": 128, "replications": 1},
            {"n": True},
            {"n": 128, "seed": -1},
        ):
            with self.assertRaises(ValueError):
                estimate("mc", **kwargs)
        with self.assertRaises(ValueError):
            estimate("qae", 128)


if __name__ == "__main__":
    unittest.main()
