# Machine Learning Portfolio

Sennai Kahssai  
Machine Learning · Scientific Computing · Applied Modeling

## Overview

This repository contains my applied machine learning portfolio, focused on building rigorous, reproducible models for quantitative and scientific problem-solving.

The work emphasizes machine learning as a modeling framework rather than a black box, integrating statistical foundations, clean data pipelines, and principled evaluation into end-to-end analytical workflows.

## Repository Structure

**projects/**
The currently checked-in notebooks are under `optimization/`:

- `optimization/` — three runnable optimization notebooks

Forecasting and learning projects are not present in this checkout yet.
Add those directories when their project notebooks are ready to include.

## Reproducibility and notebook checks

Create the shared environment from the repository root:

```bash
conda env create -f environment.yml
conda activate ml-projects
python .github/scripts/execute_notebooks.py
```

GitHub Actions executes the offline optimization notebooks on pushes and pull requests. Notebooks that require local data or external services should only be added to automatic execution once those inputs are available safely.

**resources/**  
Supporting references and methodological notes.

## Orientation

The focus is on interpretable, well-validated models designed for real-world analytical and scientific applications.
---
## Connect With Me
- [LinkedIn](https://www.linkedin.com/in/senay-kahsay)
- [Medium](https://medium.com/@kahssai.sen)

> “We are the music while the music lasts.” — *T. S. Eliot*
