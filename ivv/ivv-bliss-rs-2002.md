# IV&V Report: bliss-rs-julia-ebook vs NASA/CR-2002-211652

**Reference:** Altus, T.D. (2002). *A Response Surface Methodology for Bi-Level
Integrated System Synthesis (BLISS)*. NASA/CR-2002-211652. May 2002.

**Code reviewed:** `/Users/troyaltus/ebooks/bliss-rs-julia-ebook/src/`  
**Date:** 2026-05-15  
**Status:** Partial — RSM machinery correct; BB architecture simplified

---

## Verification Matrix

| Item | Eq/Sec | Code location | Status | Notes |
|---|---|---|---|---|
| NS minimum points | Eq. 14 | `bliss_rs.jl:20` | ✓ PASS | Exact match |
| OLS coefficients | Eq. 10 | `bliss_rs.jl:51` | ✓ IMPROVED | `\` (QR) vs explicit inversion — more numerically stable |
| Design matrix [X] | Eq. 13, §4.1.2 | `bliss_rs.jl:30–41` | ⚠ MINOR | Column ordering differs: paper separates pure quadratic then cross terms; Julia interleaves via `for i, j in i:n`. Math equivalent — β indices don't match paper's notation |
| Coded/natural mapping | Eq. 17 | `bliss_rs.jl:243,250` | ✓ PASS | Exact match both directions |
| Interval reduction A | Eq. 18 | `bliss_rs.jl:267–270` | ✓ PASS | `A = |C − opt| / half-width` ✓ |
| Interval reduction R | Eq. 19 | `bliss_rs.jl:271` | ✓ PASS | `R = A + (1−A)·K` ✓ |
| Interval clipping to physical bounds | §6 prose | `bliss_rs.jl:273–274` | ✓ PASS | `max(lo,…)`, `min(hi,…)` ✓ |
| Error metric | Eq. 16 | `bliss_rs.jl:71–75` | ✓ IMPROVED | Paper: signed point-wise. Julia: mean absolute + near-zero floor + LOOCV |
| Convergence criterion | §7 prose | `bliss_rs.jl:305–328` | ✓ IMPROVED | Paper: fixed 10 iterations. Julia: formal ε_obj + ε_Z relative tolerances |
| Hypersphere antibunching | App. A | `bliss_rs.jl:151–159` | ⚠ INCOMPLETE | Antibunching not implemented. Low risk: paper abandoned hypersphere DOE; LHS is the active method |
| **BB architecture — local optimization** | §3.1, Eq. 2, Table 2 | `supersonic_jet.jl` | ❌ SIMPLIFIED | Paper BBs take Z, Y*, w as inputs then locally optimize X. Julia BBs are direct analysis functions — no inner optimization, no w variables |
| **Weighting factors w** | §3.1, Table 2 | `bliss_rs_iteration` | ❌ SIMPLIFIED | w-variables are design variables at BB level in paper (11–14 inputs per BB). Julia drops w entirely — single-level not two-level decomposition |
| Breguet range equation | §2.2 | `supersonic_jet.jl:148–160` | ✓ PASS | `R = (a·M/SFC)·(L/D)·ln(Wᵢ/Wf)` correct |
| DOE point counts | Table 2 | N/A | ⚠ CANNOT VERIFY | Paper: 78/120/21/28 points per BB. Julia: generic `NS + n_doe_extra` not BB-specific |
| **Fig 8.1 true optimum** | N/A (pedagogical demo) | `06-interval-reduction.qmd:142` | ❌ BUG | `x_true_opt = 0.65` hardcoded. Actual minimum of `(x−0.65)²+0.1·sin(5x)` is x*≈0.8068 — sine term shifts it. Dashed reference line was wrong. Fixed 2026-05-15. |
| ELAPS structures model | §2.2, Figs 3–6 | `supersonic_jet.jl:47–62` | ❌ SIMPLIFIED | ELAPS Fortran FEM replaced by empirical Raymer weight formula + linear twist model |
| AWAVE aerodynamics | §2.2, Fig 7 | `supersonic_jet.jl:77–99` | ❌ SIMPLIFIED | AWAVE Fortran CFD replaced by analytic drag polar. Not area-ruling CFD |

---

## Critical Findings (❌)

### 1. BB architecture — no inner local optimization (highest priority)

Paper's BLISS-RS loop (Fig. 10, §3.1): for each DOE point (Z, Y*, w), locally optimize
X to minimize F_k = Σwⱼ·Yⱼ^, record optimized Yⱼ^, fit RS through those optimized outputs.

Julia's `bliss_rs_iteration` calls `bb(pts_natural[j, :])` directly — no inner
optimization. This is single-level surrogate optimization, not BLISS-RS as defined.

**Impact:** For the simplified BBs (no real X variables), difference collapses harmlessly.
But the claim "implements BLISS-RS" is architecturally inaccurate. Convergence behavior
and final optimum may differ from the paper's results.

### 2. Weighting factors w absent

Table 2 shows each BB takes 2 w-variables as additional DOE inputs (11–14 total inputs
per BB). These are determined by the system optimizer. Without w, the coupling between
BB local objectives and system objective is lost — the mechanism that makes BLISS-RS
a decomposition method vs a monolithic surrogate.

### 4. Fig 8.1 — incorrect true optimum reference line

`06-interval-reduction.qmd` used `x_true_opt = 0.65` as the "true optimum" of the
1D demo function `f(x) = (x − 0.65)² + 0.1·sin(5x)`. This is incorrect: x = 0.65
minimizes only the quadratic term. The sine term is large enough to shift the true
minimum to x* ≈ 0.8068 (verified numerically: Newton convergence on f'(x) = 0,
residual < 1e-15). f(0.65) = −0.0108; f(0.8068) = −0.0533 — a factor of ~5 difference.

The algorithm was converging correctly. The dashed reference line was wrong.

**Fix:** `x_true_opt = 0.80684301` hardcoded with Newton-converged comment.
**File:** `06-interval-reduction.qmd`, lines 88, 131, 142. **Fixed 2026-05-15.**

### 3. BB physics (ELAPS, AWAVE)

Paper results (Figs 22–50) came from ELAPS structural FEM + AWAVE wave-drag CFD.
Julia uses analytic surrogates. The optimization histories in the ebook cannot be
compared to the paper's figures — different physics.

---

## Improvements Over Reference (✓ IMPROVED)

- **OLS solver:** `\` (QR) vs explicit `(X'X)⁻¹` — stable for near-singular matrices
- **Error metric:** LOOCV + mean absolute normalized error vs Eq. 16 signed point-wise
- **Convergence:** Formal ε_obj + ε_Z tolerances vs arbitrary 10-iteration stop
- **DOE:** LHS replacing D-optimal+random hybrid — cleaner, no contamination from failed local optimizations

---

## Overall Assessment

**Partial.** The RSM machinery (Eqs. 10, 13, 14, 17, 18, 19) is correctly implemented
and in several cases improved. The loop structure and interval reduction are faithful.

However, the ebook does not implement the full two-level BLISS-RS architecture:
inner BB local optimization and w-variables are absent; BB physics are analytic
surrogates. What the ebook implements is better described as **single-level quadratic
surrogate optimization on simplified BBs** — correct as a pedagogical demonstration
of RSM, but architecturally distinct from the paper.

## Implications for AIAA journal submission

Two options:

**Option A:** Implement w-variables and inner BB optimization. Requires real BB physics
(ELAPS replacement, drag polar with area ruling). Major effort.

**Option B:** Explicitly acknowledge in the paper that the Julia implementation uses
simplified BBs without local optimization. Call it a "reduced-fidelity reproduction"
and frame the primary contribution as RSM modernization (quadratic → Kriging, LHS,
formal convergence) rather than exact replication.

Option B is defensible if the GP vs quadratic RS comparison (Ch. 9) is the main claim.
