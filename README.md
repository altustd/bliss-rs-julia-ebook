# A Response Surface Methodology for Bi-Level Integrated System Synthesis (BLISS-RS)

Recreation and extension of Troy Altus's 2002 Master's thesis (NASA/CR-2002-211652) in Julia.

## View

**[https://altustd.github.io/bliss-rs-julia-ebook/](https://altustd.github.io/bliss-rs-julia-ebook/)**

## Run

```bash
git clone git@github.com:altustd/bliss-rs-julia-ebook.git
cd bliss-rs-julia-ebook
pixi run setup    # install Julia packages
pixi run render   # HTML → docs/
pixi run preview  # live preview
```

## What's Inside

| Chapter | Topic |
|---------|-------|
| 1 | Introduction: MDO decomposition and the BLISS family |
| 2 | BLISS-98: system architecture and supersonic business jet BBs |
| 3 | BLISS-RS formulation: local and system optimization |
| 4 | Response surface methodology: least squares, second-order models |
| 5 | Design of experiments: hypersphere, D-optimal, coded variables |
| 6 | Interval reduction: adaptive design space refinement |
| 7 | BLISS-RS results: supersonic business jet optimization |
| 8 | Black box fidelity improvements: CFD and flutter constraint |
| 9 | Modern extensions: Gaussian processes, Bayesian optimization, Julia parallelism |
| 10 | Application to solid propellant rocket motor design |

## Tech Stack

Julia · Plots.jl · Optim.jl · Distributions.jl · Quarto · pixi

## Original Reference

Altus, T. D., "A Response Surface Methodology for Bi-Level Integrated System
Synthesis (BLISS)," NASA/CR-2002-211652, George Washington University /
NASA Langley Research Center, Hampton, VA, May 2002.
