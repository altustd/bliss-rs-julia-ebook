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
    rs_normalized_error(y_actual, y_rs) -> Float64

Compute the mean absolute normalized error between true and RS-predicted values.
  error = |y_actual - y_rs| / |y_actual|   (Eq. 16, Altus 2002)
"""
function rs_normalized_error(y_actual::AbstractVector, y_rs::AbstractVector)
    return mean(abs.((y_actual .- y_rs) ./ y_actual))
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

    det_val(P) = det(build_X(P)' * build_X(P))
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
# 5. BLISS-RS Top-Level Loop (generic)
# ---------------------------------------------------------------------------

"""
    bliss_rs_iteration(black_boxes, system_opt!, n_vars_Z, bounds_Z;
                       n_iter=10, K=0.8, n_doe_pts_fn=min_rs_points,
                       seed=42, verbose=true)

Generic BLISS-RS iteration loop (Section 7, Altus 2002).

Arguments:
  black_boxes  : Vector of functions `(Z, Y_star, w) -> (F_local, Y_out)`
                 where Y_out are the coupling outputs of each BB.
  system_opt!  : Function `(rs_models) -> (Z_opt, Y_opt, w_opt, obj_opt)`
                 Runs the system-level optimisation over the fitted RS models.
  n_vars_Z     : Number of system-level design variables.
  bounds_Z     : (n_vars_Z × 2) matrix of initial [lo, hi] bounds.

Returns a named tuple with iteration histories.
"""
function bliss_rs_iteration(black_boxes, system_opt!, n_vars_Z::Int,
                             bounds_Z::AbstractMatrix;
                             n_iter::Int=10, K::Float64=0.8, seed::Int=42)
    bounds = copy(bounds_Z)
    Z_history  = Vector{Vector{Float64}}()
    obj_history = Float64[]

    for it in 1:n_iter
        # 1. Generate DOE points (D-Optimal) within current bounds
        n_doe = min_rs_points(n_vars_Z) + 5   # slightly over-determined
        coded = d_optimal_points(n_vars_Z, n_doe; seed=seed + it)
        # Map coded [0,1] → natural variable space
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

        # 4. Reduce intervals around optimum
        bounds = update_intervals(Z_opt, bounds; K=K)
    end

    return (Z_history=Z_history, obj_history=obj_history)
end
