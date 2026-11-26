# Fuzzy resource allocation

A two-product factory allocation model tests how triangular profit/resource uncertainty changes a linear-programming decision. The original “Carbon Dioxide Storage” title did not describe the implemented mathematics; no carbon-storage simulation is claimed here.

**Implementation.** Nine alpha-cut levels compare four explicit policies: the original midpoint-profit/lower-resource policy, robust lower-bound profit/resources, fixed nominal values and optimistic upper bounds. For nonnegative production, lower resources ensure feasibility throughout the uncertainty box, and lower profits define its worst-case objective. HiGHS solves each LP; independent polygon-vertex enumeration verifies optimality and feasibility.

**Result.** All **36 cases** match the independent objective reference to the reported precision (maximum observed absolute gap: **0.0**). Robust objective profit rises from **90.45 at alpha 0** to **105.0 at alpha 1**; the nominal policy remains 105.0. Tests also check robust feasibility/profit at every corner of each uncertainty box.

[Executed notebook](Fuzzy_Resource_Allocation.ipynb) · [Records](results/results.json) · [Figure](results/benchmark.svg)

```bash
python -m classical_studies.run fuzzy --output artifacts/fuzzy
```

Alpha is a fuzzy membership level, not a confidence level or event probability. Resource/product values are synthetic. Reported profits use each policy's coefficients, so the figure compares policy-dependent objectives rather than measured realized returns. No commercial solver licence is required.
