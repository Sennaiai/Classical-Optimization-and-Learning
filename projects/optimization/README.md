# Optimization

| Study or reusable solver | Focus |
|---|---|
| [Nonconvex optimization](nonconvex/) | Gradient descent and BFGS on Rosenbrock/Himmelblau, checked against analytic and numerical references. |
| [Multiobjective unit commitment](unit_commitment/) | Matched-budget MOPSO/QPSO with a known Pareto front and independent feasibility audits. |
| [Fuzzy resource allocation](fuzzy_allocation/) | Robust allocation under triangular profit/resource uncertainty. |
| [Classical QUBO/Ising allocation library](../../src/classical_studies/baselines/allocation/) | Simulated annealing, exact small references, capacity-aware repair and route refinement. |

The allocation library is extracted from the [airline-recovery study](https://github.com/Sennaiai/Quantum-Computing/tree/main/projects/QUBO%20IROPS%20solver); its classical implementation runs independently here. Each notebook retains its own results, references and limitations.
