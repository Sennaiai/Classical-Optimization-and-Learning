# Monte Carlo option pricing

Matched-budget numerical estimation for European calls under risk-neutral geometric Brownian motion. The two finance notebooks are consolidated into one classical study; an unused quantum circuit and a classical dot product previously labeled as quantum estimation are excluded.

**Protocol.** Plain Monte Carlo, antithetic normals and independently scrambled Sobol/Halton sequences use identical total payoff-evaluation budgets. Antithetic pairs count as two payoffs. Each estimate averages 16 independent randomizations; four fixed outer seeds, four budget levels and three moneyness settings produce **192 records**. Replication-based Student-t intervals are approximate; no universal coverage or convergence rate is claimed. Black–Scholes supplies an independent analytic price, including zero-maturity/volatility limits.

**Result.** The standard at-the-money reference is **10.4505835722**. At the largest budget (131,072 payoff evaluations per estimate), the saved table reports mean price, absolute-error RMSE across four seeds and replication standard error for every method/case. Scrambled quasi-Monte Carlo improves observed accuracy on these one-dimensional integrals; this small experiment does not establish superiority for general derivatives.

[Executed notebook](Monte_Carlo_Option_Pricing.ipynb) · [Records](results/results.json) · [Figure](results/benchmark.svg)

```bash
python -m classical_studies.run option_pricing --output artifacts/option_pricing
```

This is statistical estimation, not a trained predictive model or market forecast. Assumptions include constant volatility, zero dividends, synthetic contracts and idealized GBM dynamics. SciPy provides the [Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) and [Halton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html) sequence generators.
