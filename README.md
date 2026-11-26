# Classical Optimization and Learning

Senay Kahsay · Scientific computing and reproducible numerical studies

## Reusable learning models

| Models | Standalone implementation | Connection |
|---|---|---|
| Ridge, random forest, gradient boosting | [Forecasting models](src/classical_studies/baselines/forecasting/models.py) and [training/prediction API](src/classical_studies/baselines/forecasting/api.py) | Classical controls for electricity-grid quantum-reservoir benchmarking. |
| Echo-state reservoir with ridge readout | [Reservoir features](src/classical_studies/baselines/forecasting/echo.py) | Classical reservoir and hybrid forecasting controls. |
| Logistic classification: full or selected molecular counts | [Molecular learning API](src/classical_studies/baselines/molecular.py) | Classical ADMET controls, scaffold-separated evaluation and portable JSON inference. |

These Python modules run independently of quantum backends. [Learning guide](projects/learning/) · [Library usage and provenance](src/classical_studies/baselines/)

## Numerical studies and optimization

| Study | Scientific focus |
|---|---|
| [Nonconvex optimization](projects/optimization/nonconvex/) | Armijo descent and BFGS; analytic derivatives, certified minima and SciPy references. |
| [Multiobjective unit commitment](projects/optimization/unit_commitment/) | MOPSO/QPSO; feasible schedules, matched budgets and an analytic/MILP Pareto reference. |
| [Fuzzy resource allocation](projects/optimization/fuzzy_allocation/) | Robust alpha-cut LPs checked against independent vertex enumeration. |
| [Monte Carlo option pricing](projects/learning/monte_carlo_estimation/) | Monte Carlo, antithetic sampling and scrambled Sobol/Halton against Black–Scholes. |

The [standalone allocation solver](src/classical_studies/baselines/allocation/) supplies classical QUBO/Ising annealing and audited refinement. [Optimization guide](projects/optimization/)

## Reproduce

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning,allocation,notebooks]'
python -m unittest discover -s tests -v
python .github/scripts/execute_notebooks.py --study all
```

Executed notebooks retain summaries, numerical evidence and limitations. CI tests only affected study groups. Monte Carlo is numerical estimation; predictive models live in the reusable library. Molecular component tests do not establish ADMET predictive accuracy.
