# Learning and statistical estimation

Reusable predictive models are implemented in Python modules; Monte Carlo is a separate numerical-estimation study.

| Model family | Implementation | Evaluation |
|---|---|---|
| Ridge, random forest, gradient boosting | [Forecasting models](../../src/classical_studies/baselines/forecasting/models.py) | Expanding chronological validation and a fixed held-out interval. |
| Echo-state reservoir with ridge readout | [Reservoir features](../../src/classical_studies/baselines/forecasting/echo.py) | Fixed seeds; reservoir-only and hybrid inputs. |
| Logistic classification | [Molecular API](../../src/classical_studies/baselines/molecular.py) | Training-only scaling, validation-only regularization selection and disjoint molecule/scaffold partitions. |

The forecasting models support the [quantum-reservoir study](https://github.com/Sennaiai/Quantum-Computing/tree/main/projects/Grid%20Load%20QRC%20Benchmarks); molecular classifiers support the [ADMET workflow](https://github.com/Sennaiai/Quantum-Computing/tree/main/projects/Quantum%20Molecular%20Learning%20for%20ADMET). Both are independently callable here.

```python
from classical_studies.baselines.forecasting import fit_forecaster, model_factory
from classical_studies.baselines.molecular import fit_classical, predict
```

[Installation, CLI and input contracts](../../src/classical_studies/baselines/) · [Monte Carlo pricing study](monte_carlo_estimation/)
