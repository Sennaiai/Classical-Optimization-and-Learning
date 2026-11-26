"""Matched-budget classical estimators for European calls under risk-neutral GBM."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, t


@dataclass(frozen=True)
class Contract:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    volatility: float = 0.2
    maturity: float = 1.0

    def validate(self):
        values = (self.spot, self.strike, self.rate, self.volatility, self.maturity)
        if any(
            isinstance(v, (bool, np.bool_))
            or not np.isscalar(v)
            or not np.isreal(v)
            or not np.isfinite(v)
            for v in values
        ):
            raise ValueError("Contract parameters must be finite real numbers")
        if (
            self.spot <= 0
            or self.strike < 0
            or self.volatility < 0
            or self.maturity < 0
        ):
            raise ValueError(
                "Require positive spot and nonnegative strike, volatility and maturity"
            )
        if (
            abs(self.rate * self.maturity) > 100
            or self.volatility * np.sqrt(self.maturity) > 10
        ):
            raise ValueError("Contract exceeds this numerical experiment budget")


def black_scholes(contract=Contract()):
    c = contract
    c.validate()
    if c.maturity == 0:
        return max(c.spot - c.strike, 0.0)
    discounted_strike = c.strike * np.exp(-c.rate * c.maturity)
    if c.volatility == 0 or c.strike == 0:
        return max(c.spot - discounted_strike, 0.0)
    scale = c.volatility * np.sqrt(c.maturity)
    d1 = (
        np.log(c.spot / c.strike) + (c.rate + 0.5 * c.volatility**2) * c.maturity
    ) / scale
    result = float(c.spot * ndtr(d1) - discounted_strike * ndtr(d1 - scale))
    if not np.isfinite(result):
        raise ArithmeticError("Non-finite analytic price")
    return result


def discounted_payoffs(normals, contract=Contract()):
    c = contract
    c.validate()
    z = np.asarray(normals, dtype=float)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("Expected finite one-dimensional normal samples")
    terminal = c.spot * np.exp(
        (c.rate - 0.5 * c.volatility**2) * c.maturity
        + c.volatility * np.sqrt(c.maturity) * z
    )
    result = np.exp(-c.rate * c.maturity) * np.maximum(terminal - c.strike, 0.0)
    if not np.isfinite(result).all():
        raise ArithmeticError("Non-finite simulated payoff")
    return result


def estimate(method, n, *, replications=16, seed=7, contract=Contract()):
    if method not in ("mc", "antithetic", "scrambled_sobol", "scrambled_halton"):
        raise ValueError("Unknown estimator")
    for value, lo, hi in ((n, 4, 2**20), (replications, 2, 128), (seed, 0, 2**32 - 1)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("Invalid integer simulation budget or seed")
    if n & (n - 1):
        raise ValueError(
            "Use power-of-two payoff budgets for matched Sobol comparisons"
        )
    reference = black_scholes(contract)
    estimates = []
    start = perf_counter()
    for stream in np.random.SeedSequence(seed).spawn(replications):
        rng = np.random.default_rng(stream)
        if method == "mc":
            z = rng.standard_normal(n)
        elif method == "antithetic":
            half = rng.standard_normal(n // 2)
            z = np.r_[half, -half]
        else:
            if method == "scrambled_sobol":
                engine = qmc.Sobol(d=1, scramble=True, seed=rng)
                uniforms = engine.random_base2(int(np.log2(n))).ravel()
            else:
                engine = qmc.Halton(d=1, scramble=True, seed=rng)
                uniforms = engine.random(n).ravel()
            z = ndtri(np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps))
        estimates.append(float(discounted_payoffs(z, contract).mean()))
    values = np.asarray(estimates)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(replications))
    half = float(t.ppf(0.975, replications - 1) * se)
    return {
        "method": method,
        "payoffs_per_replication": int(n),
        "replications": int(replications),
        "total_payoff_evaluations": int(n * replications),
        "seed": int(seed),
        "contract": asdict(contract),
        "price": mean,
        "analytic_price": reference,
        "absolute_error": abs(mean - reference),
        "replicate_standard_error": se,
        "approximate_95_percent_interval": [mean - half, mean + half],
        "replication_prices": estimates,
        "seconds": perf_counter() - start,
    }


def run_study():
    cases = {
        "out_of_money": Contract(spot=80.0),
        "at_money": Contract(),
        "in_money": Contract(spot=120.0),
    }
    rows = []
    for case, contract in cases.items():
        for n in (128, 512, 2048, 8192):
            for seed in (7, 19, 41, 73):
                for method in (
                    "mc",
                    "antithetic",
                    "scrambled_sobol",
                    "scrambled_halton",
                ):
                    row = estimate(method, n, seed=seed, contract=contract)
                    row["case"] = case
                    rows.append(row)
    return {
        "assumptions": "European calls, constant-volatility risk-neutral GBM, zero dividends; synthetic contracts, not market prediction.",
        "uncertainty": "Approximate Student-t intervals across independent randomizations. These are not simultaneous bands or a guarantee of empirical coverage.",
        "budget": "Each method uses identical total payoff evaluations; antithetic pairs count as two payoffs. Scrambled Sobol/Halton replicates use independent randomizations.",
        "records": rows,
    }
