# BLISS-2026

**Decomposition-Based Multidisciplinary Design Optimization — A Modern Julia Treatment**

An updated thesis extending Altus & Sobieski (2002) — *A Response Surface Methodology
for Bi-Level Integrated System Synthesis* (NASA/CR-2002-211652) — into a fully modern
framework using Gaussian-process surrogates, Bayesian optimization, and Julia.

## View

**[https://altustd.github.io/bliss2026/](https://altustd.github.io/bliss2026/)**

## Run

```bash
git clone git@github.com:altustd/bliss2026.git
cd bliss2026
pixi run setup    # install Julia packages (requires Julia 1.10+ in PATH)
pixi run render   # HTML → docs/
pixi run preview  # live preview
```

## What's Inside

| # | Chapter | Highlights |
|---|---------|-----------|
| 1 | Introduction | MDO decomposition, BLISS family, RS vs. GP |
| 2 | BLISS-98 | System architecture, supersonic BJ black boxes |
| 3 | BLISS-RS Formulation | Local/system optimization, weighting factors |
| 4 | Response Surface Methodology | Least squares, second-order models, error analysis |
| 5 | Design of Experiments | Hypersphere, D-optimal, Latin Hypercube |
| 6 | Interval Reduction | Adaptive trust-region bounds management |
| 7 | Supersonic Business Jet | BLISS-RS results, BLISS-2026 comparison |
| 8 | Black Box Fidelity | CFD integration, flutter constraint |
| **9** | **BLISS-2026 Framework** | **GP surrogates, EI acquisition, parallel Julia, full implementation** |
| 10 | SRM Application | Solid propellant rocket motor design |
| **11** | **Modern MDO Landscape** | **Multi-fidelity, neural surrogates, UQ, digital twins** |

## Core Algorithm: BLISS-2026

BLISS-2026 replaces the 2002 quadratic-polynomial surrogates with:

| Component | 2002 (BLISS-RS) | 2026 (BLISS-2026) |
|-----------|----------------|------------------|
| Surrogate | Quadratic polynomial | Gaussian Process (SE kernel) |
| Hyperparameters | Fixed by analyst | Optimised via marginal likelihood |
| Sampling | D-Optimal DOE | Latin Hypercube + EI-adaptive |
| System optimisation | Nelder-Mead over RS | Bayesian optimisation (Expected Improvement) |
| Uncertainty | None | GP predictive variance throughout |
| BB evaluation | Serial | Parallel via `Threads.@threads` |
| Convergence | Fixed 20% interval shrink | Trust-region bounds management |

## Tech Stack

Julia · Plots.jl · Optim.jl · Distributions.jl · LinearAlgebra · Quarto · pixi

## Original Reference

Altus, T. D., "A Response Surface Methodology for Bi-Level Integrated System
Synthesis (BLISS)," NASA/CR-2002-211652, George Washington University /
NASA Langley Research Center, Hampton, VA, May 2002.
