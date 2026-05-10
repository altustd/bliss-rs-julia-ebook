# =============================================================================
# bliss_rs.jl — Core BLISS-RS algorithm implementations
# Implements Altus & Sobieski (2002), NASA/CR-2002-211652
# =============================================================================

using LinearAlgebra, Statistics, Random, Printf

# ---------------------------------------------------------------------------
# 1. Response Surface Model
# ---------------------------------------------------------------------------

"""
    min_rs_points(n) -> Int

Minimum number of sample points to uniquely fit a full second-order response
surface model in `n` variables (Eq. 14 from Altus 2002).

  NS = (n² + 3n + 2) / 2
"""
min_rs_points(n::Int) = (n^2 + 3n + 2) ÷ 2

"""
    rs_design_matrix(pts) -> Matrix

Build the design matrix [X] for a second-order (quadratic) response surface.
Rows = observations; columns = [1, x₁…xₙ, x₁², x₁x₂, …, xₙ²].

Equation reference: Section 4.1.2, Altus (2002).
"""
function rs_design_matrix(pts::AbstractMatrix{<:Real})
    m, n = size(pts)
    n_cross = n * (n + 1) ÷ 2          # pure quadratic + cross terms
    X = ones(m, 1 + n + n_cross)
    X[:, 2:n+1] = pts
    col = n + 2
    for i in 1:n, j in i:n
        X[:, col] = pts[:, i] .* pts[:, j]
        col += 1
    end
    return X
end

"""
    fit_rs(pts, y) -> Vector

Fit a second-order response surface by least squares.
Returns the coefficient vector b̂ = (X'X)⁻¹X'y  (Eq. 10, Altus 2002).
"""
function fit_rs(pts::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    X = rs_design_matrix(pts)
    return (X' * X) \ (X' * y)
end

"""
    predict_rs(pts, b) -> Vector

Predict response surface values at new sample points using coefficients b.
"""
function predict_rs(pts::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    return rs_design_matrix(pts) * b
end

"""
    rs_normalized_error(y_actual, y_rs; tol=1e-8) -> Float64

Mean absolute normalized error (Eq. 16, Altus 2002), stabilized for near-zero
response values.  The original formulation divides by y_actual and diverges when
y_actual → 0 (e.g. the aeroelastic twist Θ passes through zero during optimization).
A floor of `tol * max(|y_actual|)` prevents division by near-zero.
"""
function rs_normalized_error(y_actual::AbstractVector, y_rs::AbstractVector;
                              tol::Float64=1e-8)
    denom = max.(abs.(y_actual), tol * maximum(abs.(y_actual)))
    return mean(abs.((y_actual .- y_rs) ./ denom))
end

"""
    rs_rmse(y_actual, y_rs) -> Float64

Root-mean-square error of the RS fit.
"""
rs_rmse(y_actual::AbstractVector, y_rs::AbstractVector) =
    sqrt(mean((y_actual .- y_rs).^2))

"""
    rs_r2(y_actual, y_rs) -> Float64

Coefficient of determination R² of the RS fit.
R² = 1 − SS_res / SS_tot. Returns 1.0 for a perfect fit; negative for worse-than-mean.
"""
function rs_r2(y_actual::AbstractVector, y_rs::AbstractVector)
    ss_res = sum((y_actual .- y_rs).^2)
    ss_tot = sum((y_actual .- mean(y_actual)).^2)
    ss_tot < 1e-14 && return 1.0   # constant response
    return 1.0 - ss_res / ss_tot
end

"""
    loocv_error(pts, y) -> NamedTuple

Leave-one-out cross-validation for a quadratic RS fit.  For each observation i,
fits the model on all other points, predicts at i, and records the error.

Returns (rmse, r2, mae, normalized_mae) — the standard validation metrics a
peer reviewer would expect.
"""
function loocv_error(pts::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    m = length(y)
    m < 2 && error("Need at least 2 points for LOOCV")
    y_loo = similar(y)
    idx = collect(1:m)
    for i in 1:m
        train_idx = filter(!=(i), idx)
        b = fit_rs(pts[train_idx, :], y[train_idx])
        y_loo[i] = predict_rs(pts[[i], :], b)[1]
    end
    rmse = rs_rmse(y, y_loo)
    r2   = rs_r2(y, y_loo)
    mae  = mean(abs.(y .- y_loo))
    nmae = rs_normalized_error(y, y_loo)
    return (rmse=rmse, r2=r2, mae=mae, normalized_mae=nmae, y_loo=y_loo)
end

"""
    rs_validate(pts, y) -> NamedTuple

Comprehensive RS validation: in-sample fit metrics + LOOCV out-of-sample metrics.
Use this instead of Eq. 16 from Altus (2002) for any submitted work.
"""
function rs_validate(pts::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    b       = fit_rs(pts, y)
    y_hat   = predict_rs(pts, b)
    insample = (rmse = rs_rmse(y, y_hat),
                r2   = rs_r2(y, y_hat),
                nmae = rs_normalized_error(y, y_hat))
    loo = loocv_error(pts, y)
    return (insample=insample, loocv=loo, coefficients=b)
end

# ---------------------------------------------------------------------------
# 2. Design Point Placement
# ---------------------------------------------------------------------------

"""
    hypersphere_points(n, m; seed) -> Matrix

Generate `m` quasi-uniform sample points on the surface of an n-dimensional
unit hypersphere (Appendix A, Altus 2002). Points are then mapped into the
interior unit hypercube via a radial projection.
"""
function hypersphere_points(n::Int, m::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    pts = randn(rng, m, n)
    for i in 1:m
        pts[i, :] ./= norm(pts[i, :])   # project to unit sphere
    end
    # Map from [-1,1]ⁿ (sphere surface) to [0,1]ⁿ
    return (pts .+ 1.0) ./ 2.0
end

"""
    random_points(n, m; seed) -> Matrix

Generate `m` uniform random points in [0,1]ⁿ.
"""
function random_points(n::Int, m::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    return rand(rng, m, n)
end

"""
    d_optimal_points(n, m; model=:quadratic, n_iter=500, seed) -> Matrix

Generate a D-optimal design with `m` points in n variables using the
coordinate exchange algorithm (Cook & Nachtsheim, 1980).
Maximises det(X'X) where X is the model matrix.
"""
function d_optimal_points(n::Int, m::Int;
                           model::Symbol=:quadratic,
                           n_iter::Int=500,
                           seed::Int=42)
    rng = MersenneTwister(seed)
    # Initialise with random points in [-1,1]
    pts = 2 .* rand(rng, m, n) .- 1.0
    build_X = pts -> rs_design_matrix((pts .+ 1.0) ./ 2.0)

    det_val(P) = abs(det(build_X(P)' * build_X(P)))
    best_det = det_val(pts)

    for _ in 1:n_iter
        i = rand(rng, 1:m)
        j = rand(rng, 1:n)
        old_val = pts[i, j]
        pts[i, j] = 2 * rand(rng) - 1.0
        new_det = det_val(pts)
        if new_det > best_det
            best_det = new_det
        else
            pts[i, j] = old_val    # revert
        end
    end
    return (pts .+ 1.0) ./ 2.0    # return in [0,1]
end

"""
    lhs_points(n, m; seed) -> Matrix{Float64}

Generate a Latin Hypercube Sample with `m` points in `n` variables, mapped to [0,1]ⁿ.

LHS divides each dimension into `m` equal-width strata and places exactly one
point per stratum, with random position within the stratum.  Compared to the
D-optimal+random hybrid used in Altus (2002), LHS:

  - Guarantees uniform projection onto each axis (no crowding/empty-band risk)
  - Requires no convergence criterion or fallback random points
  - Is not contaminated by D-optimality loss when local BB optimizations fail

For computer experiments (deterministic BBs) this is the standard first choice;
see Sacks et al. (1989) "Design and Analysis of Computer Experiments".
"""
function lhs_points(n::Int, m::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    pts = zeros(m, n)
    for j in 1:n
        perm = randperm(rng, m)
        for i in 1:m
            pts[i, j] = (perm[i] - 1 + rand(rng)) / m
        end
    end
    return pts
end

# ---------------------------------------------------------------------------
# 3. Coded ↔ Natural Variable Mapping
# ---------------------------------------------------------------------------

"""
    code_to_natural(ξ, xL, xU) -> Real

Map coded variable ξ ∈ [-1, 1] to natural (physical) variable x ∈ [xL, xU].
  x = xL + (ξ + 1)/2 * (xU - xL)   (Eq. 17, Altus 2002)
"""
code_to_natural(ξ, xL, xU) = xL + (ξ + 1) / 2 * (xU - xL)

"""
    natural_to_code(x, xL, xU) -> Real

Map physical variable x ∈ [xL, xU] to coded variable ξ ∈ [-1, 1].
"""
natural_to_code(x, xL, xU) = 2 * (x - xL) / (xU - xL) - 1

# ---------------------------------------------------------------------------
# 4. Interval Reduction
# ---------------------------------------------------------------------------

"""
    interval_reduction_factors(opt_val, bounds_lo, bounds_hi, K=0.8)
        -> (new_lo, new_hi)

Compute new interval bounds centred on the current optimum with a contraction
factor that depends on proximity to the boundary (Eqs. 18–19, Altus 2002).

  Aᵢ = |Cᵢ - optᵢ| / (|Cᵢ - Lᵢ| + 1e-12)
  Rᵢ = Aᵢ + (1 - Aᵢ) * K
  new_width = Rᵢ * (Hᵢ - Lᵢ)
"""
function interval_reduction_factors(opt::Real, lo::Real, hi::Real; K::Float64=0.8)
    centre = (lo + hi) / 2.0
    half   = (hi - lo) / 2.0 + 1e-12
    A      = abs(centre - opt) / half
    R      = A + (1.0 - A) * K
    new_half = R * (hi - lo) / 2.0
    new_lo   = max(lo, opt - new_half)   # respect physical bounds
    new_hi   = min(hi, opt + new_half)
    return new_lo, new_hi
end

"""
    update_intervals(Z_opt, bounds; K=0.8) -> Matrix

Apply interval reduction to all system variables.
`bounds` is an (n × 2) matrix of [lo, hi] per variable.
Returns an updated (n × 2) bounds matrix.
"""
function update_intervals(Z_opt::AbstractVector, bounds::AbstractMatrix;
                          K::Float64=0.8)
    n = length(Z_opt)
    new_bounds = similar(bounds)
    for i in 1:n
        lo, hi = bounds[i, 1], bounds[i, 2]
        new_bounds[i, 1], new_bounds[i, 2] =
            interval_reduction_factors(Z_opt[i], lo, hi; K=K)
    end
    return new_bounds
end

# ---------------------------------------------------------------------------
# 5. Convergence Criteria
# ---------------------------------------------------------------------------

"""
    ConvergenceCriteria

Formal stopping conditions for the BLISS-RS iteration loop.

Fields:
  ε_obj     Relative objective change:  |f_{i+1} - f_i| / (|f_i| + 1e-10) < ε_obj
  ε_Z       Relative design change:     ||Z_{i+1} - Z_i||₂ / (||Z_i||₂ + 1e-10) < ε_Z
  max_iter  Hard iteration cap (safety net; convergence may be declared earlier)

The original Altus (2002) used a fixed 10-iteration count with visual inspection
of convergence plots. That is not a defensible stopping criterion: convergence was
declared by the approaching thesis deadline rather than by any quantitative test.
These criteria replace it.

Typical values for engineering MDO problems:
  ε_obj = 1e-4   (0.01% objective change)
  ε_Z   = 1e-4   (0.01% design variable change)
  max_iter = 50
"""
struct ConvergenceCriteria
    ε_obj::Float64
    ε_Z::Float64
    max_iter::Int
end

ConvergenceCriteria(; ε_obj=1e-4, ε_Z=1e-4, max_iter=50) =
    ConvergenceCriteria(ε_obj, ε_Z, max_iter)

"""
    check_convergence(Z_new, Z_old, obj_new, obj_old, tol) -> Bool

Return true if all active convergence criteria are satisfied.
"""
function check_convergence(Z_new, Z_old, obj_new, obj_old, tol::ConvergenceCriteria)
    Δobj = abs(obj_new - obj_old) / (abs(obj_old) + 1e-10)
    ΔZ   = norm(Z_new .- Z_old) / (norm(Z_old) + 1e-10)
    return Δobj < tol.ε_obj && ΔZ < tol.ε_Z
end

# ---------------------------------------------------------------------------
# 6. BLISS-RS Top-Level Loop (generic)
# ---------------------------------------------------------------------------

"""
    bliss_rs_iteration(black_boxes, system_opt!, n_vars_Z, bounds_Z; ...)

Generic BLISS-RS iteration loop (Section 7, Altus 2002), with formal convergence
criteria replacing the original fixed-iteration termination.

Arguments:
  black_boxes  : Vector of functions `(pts_row::Vector) -> scalar`
                 Each function evaluates one BB output at a design point.
  system_opt!  : Function `(rs_models, bounds) -> (Z_opt::Vector, obj_opt::Float64)`
                 Runs the system-level optimisation over the fitted RS models.
  n_vars_Z     : Number of system-level design variables.
  bounds_Z     : (n_vars_Z × 2) matrix of initial [lo, hi] bounds.

Keyword arguments:
  tol          : ConvergenceCriteria (default ε_obj=1e-4, ε_Z=1e-4, max_iter=50)
  K            : Interval contraction factor (Altus 2002 used 0.8)
  n_doe_extra  : Sample points above the quadratic minimum NS (default 5)
  seed         : RNG seed for reproducibility

Returns a named tuple:
  Z_history    : Vector of Z_opt per iteration
  obj_history  : Objective value per iteration
  Δobj_history : Relative objective change per iteration (for convergence plots)
  ΔZ_history   : Relative design change per iteration
  converged    : Bool — true if criteria met before max_iter
  n_iter       : Actual number of iterations run
"""
function bliss_rs_iteration(black_boxes, system_opt!, n_vars_Z::Int,
                             bounds_Z::AbstractMatrix;
                             tol::ConvergenceCriteria = ConvergenceCriteria(),
                             K::Float64=0.8,
                             n_doe_extra::Int=5,
                             seed::Int=42)
    bounds       = copy(bounds_Z)
    Z_history    = Vector{Vector{Float64}}()
    obj_history  = Float64[]
    Δobj_history = Float64[]
    ΔZ_history   = Float64[]
    converged    = false

    for it in 1:tol.max_iter
        # 1. Generate DOE points (LHS) within current bounds.
        # LHS replaces the D-optimal+random-fallback hybrid from Altus (2002).
        n_doe = min_rs_points(n_vars_Z) + n_doe_extra
        coded = lhs_points(n_vars_Z, n_doe; seed=seed + it)
        pts_natural = similar(coded)
        for i in 1:n_vars_Z
            lo, hi = bounds[i, 1], bounds[i, 2]
            pts_natural[:, i] = lo .+ coded[:, i] .* (hi - lo)
        end

        # 2. Evaluate each BB at DOE points, fit quadratic RS
        rs_models = Dict{Int, Vector{Float64}}()
        for (k, bb) in enumerate(black_boxes)
            y_samples = [bb(pts_natural[j, :]) for j in 1:n_doe]
            rs_models[k] = fit_rs(coded, y_samples)
        end

        # 3. System optimisation over RS models
        Z_opt, obj_opt = system_opt!(rs_models, bounds)

        push!(Z_history, Z_opt)
        push!(obj_history, obj_opt)

        # 4. Convergence check (requires at least 2 iterations)
        if it > 1
            Δobj = abs(obj_opt - obj_history[end-1]) / (abs(obj_history[end-1]) + 1e-10)
            ΔZ   = norm(Z_opt .- Z_history[end-1]) / (norm(Z_history[end-1]) + 1e-10)
            push!(Δobj_history, Δobj)
            push!(ΔZ_history, ΔZ)
            if check_convergence(Z_opt, Z_history[end-1], obj_opt,
                                  obj_history[end-1], tol)
                converged = true
                break
            end
        else
            push!(Δobj_history, NaN)
            push!(ΔZ_history, NaN)
        end

        # 5. Reduce intervals around optimum
        bounds = update_intervals(Z_opt, bounds; K=K)
    end

    return (Z_history    = Z_history,
            obj_history  = obj_history,
            Δobj_history = Δobj_history,
            ΔZ_history   = ΔZ_history,
            converged    = converged,
            n_iter       = length(obj_history))
end
