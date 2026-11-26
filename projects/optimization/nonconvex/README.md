# Nonconvex optimization

Rosenbrock's narrow valley and Himmelblau's multiple minima test how first- and second-order methods behave under identical accuracy criteria. This study consolidates the two original fundamentals notebooks.

**Implementation.** Analytic objectives, gradients and Hessians; Armijo backtracking gradient descent; inverse-BFGS with curvature safeguards; independent SciPy BFGS. Four Rosenbrock starts and five Himmelblau starts are fixed before execution. Termination status is reported separately from scientific acceptance: objective ≤ 10⁻¹⁰, gradient infinity norm ≤ 10⁻⁶ and distance to a certified global minimum ≤ 10⁻⁵. Function/gradient counts cover optimizer calls; final reference diagnostics are additional.

**Result.** All **27/27 runs** meet these criteria. BFGS requires substantially fewer iterations than gradient descent on Rosenbrock; evaluation counts accompany iteration counts. Himmelblau's four zero-residual roots are independently derived from its quartic, with positive Hessian eigenvalues checked. Finite differences validate both derivative orders.

[Executed notebook](Nonconvex_Optimization.ipynb) · [Numerical records](results/results.json) · [Figure](results/benchmark.svg)

```bash
python -m classical_studies.run nonconvex --output artifacts/nonconvex
```

These are two-dimensional deterministic benchmarks with a declared start set; they do not establish global convergence from arbitrary initial points. The figure samples one fixed start per objective; the table retains every run. SciPy contributes a final-point marker rather than an invented iteration history.
