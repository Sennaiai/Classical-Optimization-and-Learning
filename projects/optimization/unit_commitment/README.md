# Multiobjective unit commitment

A controlled MOPSO–QPSO comparison on the original synthetic two-unit, four-hour system. Cost and emissions include startup contributions. QPSO is a classical randomized optimizer inspired by a quantum-behavior model.

**Model and protocol.** Demand is [60, 90, 110, 70] MW. Explicit commitment/startup variables enforce exact hourly balance, minimum/maximum output and initially-off startup logic. A nearest-feasible projection replaces the original subminimum dispatch and implicitly weighted repair. Both methods share archive, leader and personal-best policies: 30 particles, 150 iterations, archive capacity 64 and five fixed seeds, giving **4,530 candidate evaluations per run**. Internal rescoring/audits are excluded from that candidate count.

**Independent reference.** Total A generation spans 50–320 MWh. Single-start commitment windows cover this interval while B stays committed (its minimum is zero). Thus the complete Pareto line is C = 13,310 − 20A and E = 35.5 + 0.8A. Seventeen epsilon-constrained HiGHS MILPs verify it. Hypervolume uses fixed ideal/nadir coordinates and reference point (1.1, 1.1); the continuous-front hypervolume is 0.71.

**Result.** All retained schedules pass an independent feasibility audit. Median hypervolume divided by the continuous reference is **0.98873 for MOPSO** and **0.95181 for QPSO**. This measures tradeoff coverage; it is neither an allocation rate nor a universal optimizer ranking.

[Executed notebook](Multiobjective_Unit_Commitment.ipynb) · [Records](results/results.json) · [Figure](results/benchmark.svg)

```bash
python -m classical_studies.run unit_commitment --output artifacts/unit_commitment
```

No ramping, minimum up/down times, reserves or network constraints are modeled. This is an explicitly adapted multiobjective QPSO experiment. Algorithm references: [MOPSO](https://doi.org/10.1109/TEVC.2004.826067), [QPSO](https://doi.org/10.1109/CEC.2004.1330875).
