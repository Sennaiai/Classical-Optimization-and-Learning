# Standalone classical baselines

These are the classical controls used in the quantum repository, extracted into
an independent package. No quantum repository checkout, quantum backend, API
credential or proprietary measurement output is required. Source commit/file
hashes and adaptations are recorded in `provenance.json`.

| Module | Models and retained protocol |
|---|---|
| `forecasting` | Persistence, daily/weekly seasonal forecasts, standardized ridge, random forest, histogram gradient boosting and classical echo-state features; chronological validation and a fixed test interval. |
| `molecular` | Full-count and selected/scaled-count logistic classifiers; training-only scaling, validation-only regularization choice, disjoint molecule/scaffold partitions and portable JSON inference. |
| `allocation` | Classical QUBO/Ising simulated annealing, exact small-instance references, capacity-ledger repair, route expansion and local allocation refinement. |

Install from the repository root:

```bash
python -m pip install -e '.[learning,allocation,notebooks]'
```

```python
from classical_studies.baselines.forecasting import model_factory, fit_forecaster
from classical_studies.baselines.molecular import fit_classical, predict
from classical_studies.baselines.allocation import solve_decomposed, score
```

The forecasting factory returns independently fit/callable sklearn estimators;
`fit_forecaster` preserves the source study's causal persistence-residual protocol.
Molecular callers supply already prepared count matrices, ordered feature names
and canonical scaffold keys. Selected-feature preprocessing must itself be fit
on training data only. This library does not claim to infer molecular identity or
validate a caller's scaffold assignment from numeric matrices.

```bash
python -m classical_studies.baselines.allocation --catalog fixture
python -m classical_studies.baselines.forecasting --output artifacts/forecasting.json
python -m classical_studies.baselines.molecular specification.json --output artifacts/model.json
```

The bundled allocation catalog is the authorized de-identified public catalog.
Forecasting data retain their NESO licence/provenance. Original QRC collaboration
credits remain in the linked source project. Molecular synthetic tests establish
software behavior, not ADMET predictive accuracy. Classical chemistry solvers and
plain library linear-system reference calls remain with their domain applications.
