# Classical Optimization and Learning

Senay Kahsay · Scientific computing and reproducible numerical studies

Four focused studies connect optimization and statistical estimation with independent references, explicit computational budgets and saved results.

| Study | Implementation and evidence |
|---|---|
| [Nonconvex optimization](projects/optimization/nonconvex/) | Armijo gradient descent and BFGS; analytic derivatives, certified minima and SciPy comparisons. |
| [Multiobjective unit commitment](projects/optimization/unit_commitment/) | MOPSO and classical QPSO; feasible schedules, equal candidate budgets and an analytic/MILP Pareto reference. |
| [Fuzzy resource allocation](projects/optimization/fuzzy_allocation/) | Robust alpha-cut linear programs; HiGHS checked against independent vertex enumeration. |
| [Monte Carlo option pricing](projects/learning/monte_carlo_estimation/) | Monte Carlo, antithetic sampling and scrambled Sobol/Halton; matched budgets and Black–Scholes references. |

[Standalone classical models](src/classical_studies/baselines/) provide forecasting, molecular logistic classification and QUBO/Ising allocation APIs extracted from the [quantum studies](https://github.com/Sennaiai/Quantum-Computing). They run independently of quantum backends. The finance study is numerical estimation; predictive learning lives in these reusable modules.

## Reproduce

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning,allocation,notebooks]'
python -m unittest discover -s tests -v
python .github/scripts/execute_notebooks.py --study all
```

Notebooks contain executed summaries; each study links its numerical evidence and limitations. CI tests and executes only affected study groups. These bounded demonstrations establish implementation behavior within their stated models.
