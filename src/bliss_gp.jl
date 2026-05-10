# =============================================================================
# bliss_gp.jl — BLISS-2026 Core: GP Surrogates + Bayesian Optimisation
#
# Replaces the quadratic polynomial surrogates of BLISS-RS (Altus 2002) with
# Gaussian Process surrogates and Expected Improvement acquisition.
#
# Key algorithmic changes vs. BLISS-RS:
#   Surrogate    : SE-kernel GP (interpolating, probabilistic) vs. quadratic poly
#   Hyperparams  : Marginal-likelihood optimisation vs. analyst-fixed
#   Sampling     : Latin Hypercube + EI-adaptive vs. D-Optimal fixed budget
#   System opt.  : Bayesian optimisation (EI) vs. Nelder-Mead over polynomial
#   Uncertainty  : Full predictive distribution vs. point estimate
#   Parallelism  : Threads.@threads for concurrent BB evaluation
#   Convergence  : Trust-region bounds update vs. fixed 20% shrinkage
# =============================================================================

using LinearAlgebra, Statistics, Distributions, Random, Printf, Optim

# ---------------------------------------------------------------------------
# 1. Kernel Functions
# ---------------------------------------------------------------------------

"""
    SEKernel(ℓ, σ²)

Squared Exponential (radial basis function) kernel:
  k(x, x') = σ² · exp(−‖x − x'‖² / (2ℓ²))

`ℓ` is the length scale; `σ²` is the signal variance.
"""
struct SEKernel
    ℓ::Float64
    σ²::Float64
end

(k::SEKernel)(x, y) = k.σ² * exp(-0.5 * sum((x .- y).^2) / k.ℓ^2)

"""
    ARDKernel(ℓ, σ²)

Automatic Relevance Determination kernel — per-dimension length scales:
  k(x, x') = σ² · exp(−½ Σᵢ (xᵢ − xᵢ')² / ℓᵢ²)
"""
struct ARDKernel
    ℓ::Vector{Float64}
    σ²::Float64
end

(k::ARDKernel)(x, y) = k.σ² * exp(-0.5 * sum(((x .- y) ./ k.ℓ).^2))

"""
    Matern52Kernel(ℓ, σ²)

Matérn 5/2 kernel — smoother than exponential, rougher than SE:
  k(r) = σ² (1 + √5·r/ℓ + 5r²/(3ℓ²)) · exp(−√5·r/ℓ)

Preferred when the true function has moderate roughness.
"""
struct Matern52Kernel
    ℓ::Float64
    σ²::Float64
end

function (k::Matern52Kernel)(x, y)
    r = sqrt(sum((x .- y).^2))
    z = sqrt(5) * r / k.ℓ
    return k.σ² * (1 + z + z^2/3) * exp(-z)
end

# ---------------------------------------------------------------------------
# 2. GP Model Struct and Fitting
# ---------------------------------------------------------------------------

"""
    GPModel

Fitted Gaussian Process surrogate. Stores training data, kernel, and the
pre-computed Cholesky factor + weight vector needed for fast prediction.
"""
struct GPModel{K}
    X_train::Matrix{Float64}   # n × d training inputs (coded [0,1]^d)
    y_train::Vector{Float64}   # n training outputs
    kernel::K
    σ_n::Float64               # noise standard deviation
    L::LowerTriangular{Float64, Matrix{Float64}}   # Cholesky of K + σ_n²I
    α::Vector{Float64}         # K⁻¹ y
end

"""
    build_kernel_matrix(X, kernel; σ_n) -> Matrix

Construct the n×n kernel matrix K(X,X) + σ_n² I.
"""
function build_kernel_matrix(X::Matrix{Float64}, kernel, σ_n::Float64)
    n = size(X, 1)
    K = [kernel(X[i,:], X[j,:]) for i in 1:n, j in 1:n]
    K += (σ_n^2 + 1e-9) * I   # jitter for numerical stability
    return Symmetric(K)
end

"""
    fit_gp(X_train, y_train, kernel; σ_n) -> GPModel

Fit a GP surrogate by exact inference (Cholesky factorisation).
Time complexity: O(n³) for fitting, O(n) for prediction mean, O(n²) for variance.
"""
function fit_gp(X_train::Matrix{Float64}, y_train::Vector{Float64},
                kernel; σ_n::Float64=1e-3)
    K  = build_kernel_matrix(X_train, kernel, σ_n)
    C  = cholesky(K)
    α  = C \ y_train
    return GPModel(X_train, y_train, kernel, σ_n, C.L, α)
end

# ---------------------------------------------------------------------------
# 3. GP Prediction
# ---------------------------------------------------------------------------

"""
    predict_gp(gp, X_test) -> (μ, σ²)

Return posterior mean μ and variance σ² at test points X_test.

  μ(x*)  = k(x*, X) α
  σ²(x*) = k(x*,x*) − k(x*,X) K⁻¹ k(X,x*)
"""
function predict_gp(gp::GPModel, X_test::Matrix{Float64})
    n_test  = size(X_test, 1)
    n_train = size(gp.X_train, 1)

    # Cross-covariance  k_s[i,j] = k(x*_i, x_j)
    k_s = [gp.kernel(X_test[i,:], gp.X_train[j,:])
           for i in 1:n_test, j in 1:n_train]

    μ = k_s * gp.α

    # Variance:  v = L⁻¹ kₛᵀ,  σ² = diag(kₛₛ) − colnorm²(v)
    v   = gp.L \ k_s'
    k_ss = [gp.kernel(X_test[i,:], X_test[i,:]) for i in 1:n_test]
    σ²  = max.(k_ss .- vec(sum(v.^2, dims=1)), 0.0)

    return μ, σ²
end

# Single-point prediction (convenience wrapper)
function predict_gp(gp::GPModel, x::Vector{Float64})
    μ, σ² = predict_gp(gp, reshape(x, 1, :))
    return μ[1], σ²[1]
end

# ---------------------------------------------------------------------------
# 4. Log Marginal Likelihood and Hyperparameter Optimisation
# ---------------------------------------------------------------------------

"""
    log_marginal_likelihood(X, y; ℓ, σ², σ_n) -> Float64

Compute the GP log marginal likelihood for an SE kernel with given hyperparameters:
  log p(y|X,θ) = −½ yᵀ K⁻¹ y − ½ log|K| − (n/2) log 2π

Used to optimise kernel hyperparameters by maximum marginal likelihood.
"""
function log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64};
                                  ℓ::Float64, σ²::Float64, σ_n::Float64=1e-3)
    n = size(X, 1)
    K = build_kernel_matrix(X, SEKernel(ℓ, σ²), σ_n)
    C = cholesky(K)
    α = C \ y
    lml = -0.5 * dot(y, α) - sum(log.(diag(C.L))) - 0.5n * log(2π)
    return lml
end

"""
    optimise_gp_hyperparams(X, y; σ_n, verbose) -> SEKernel

Maximise the log marginal likelihood over log(ℓ) and log(σ²) using L-BFGS.
Returns the best-fit SEKernel for further GP construction.
"""
function optimise_gp_hyperparams(X::Matrix{Float64}, y::Vector{Float64};
                                   σ_n::Float64=1e-3, verbose::Bool=false)
    y_std = std(y) + 1e-8
    y_norm = (y .- mean(y)) ./ y_std   # normalise for numerical stability

    # Optimise in log space; initialise at data-informed values
    ℓ_init  = 0.25 * mean(std(X, dims=1))   # ~ quarter of variable range
    σ²_init = var(y_norm) + 0.1
    θ_init  = [log(max(ℓ_init, 1e-4)), log(max(σ²_init, 1e-4))]

    neg_lml(θ) = -log_marginal_likelihood(X, y_norm;
                                           ℓ=exp(θ[1]), σ²=exp(θ[2]), σ_n=σ_n/y_std)

    result = Optim.optimize(neg_lml, θ_init, LBFGS(),
                             Optim.Options(iterations=300, show_trace=verbose,
                                           g_tol=1e-5))

    ℓ_opt, σ²_opt = exp.(Optim.minimizer(result))
    # σ²_opt is in normalised-y space; keep it there — fit_gp_auto fits to y_norm
    return SEKernel(ℓ_opt, σ²_opt), mean(y), y_std
end

"""
    fit_gp_auto(X_train, y_train; σ_n) -> GPModel

Fit a GP with automatic hyperparameter optimisation via marginal likelihood.
Returns a GP fitted to the normalised outputs but predicts on the original scale.
"""
function fit_gp_auto(X_train::Matrix{Float64}, y_train::Vector{Float64};
                      σ_n::Float64=1e-3)
    kernel, y_mean, y_std = optimise_gp_hyperparams(X_train, y_train; σ_n=σ_n)
    y_norm = (y_train .- y_mean) ./ y_std
    gp_norm = fit_gp(X_train, y_norm, kernel; σ_n=σ_n/y_std)
    # Wrap so predictions are on original scale
    return gp_norm, y_mean, y_std
end

# Predict on original scale (un-normalise)
function predict_gp_scaled(gp_norm::GPModel, y_mean::Float64, y_std::Float64,
                             X_test::Matrix{Float64})
    μ_n, σ²_n = predict_gp(gp_norm, X_test)
    return μ_n .* y_std .+ y_mean, σ²_n .* y_std^2
end

# ---------------------------------------------------------------------------
# 5. Acquisition Functions
# ---------------------------------------------------------------------------

"""
    expected_improvement(μ, σ², f_best; ξ) -> Vector

Expected Improvement for **maximisation**:
  EI(x) = (μ(x) − f* − ξ) Φ(z) + σ(x) φ(z)
  z      = (μ(x) − f* − ξ) / σ(x)

ξ > 0 is an exploration bonus (default 0.01).
EI is zero when σ = 0 (already observed point).
"""
function expected_improvement(μ::Vector{Float64}, σ²::Vector{Float64},
                                f_best::Float64; ξ::Float64=0.01)
    σ  = sqrt.(max.(σ², 1e-12))
    z  = (μ .- f_best .- ξ) ./ σ
    ei = (μ .- f_best .- ξ) .* cdf.(Normal(), z) .+ σ .* pdf.(Normal(), z)
    return max.(ei, 0.0)
end

"""
    upper_confidence_bound(μ, σ², β) -> Vector

UCB acquisition: UCB(x) = μ(x) + √β · σ(x)
β controls exploration strength; typical range 1–3.
"""
upper_confidence_bound(μ, σ², β=2.0) = μ .+ sqrt(β) .* sqrt.(max.(σ², 0.0))

# ---------------------------------------------------------------------------
# 6. Latin Hypercube Sampling
# ---------------------------------------------------------------------------

"""
    lhs_sample(n_vars, n_pts; seed) -> Matrix  (n_pts × n_vars, values in [0,1])

Latin Hypercube Sampling: divides each dimension into n_pts equally-spaced strata,
then randomly samples one point per stratum per dimension.

Compared to uniform random, LHS guarantees projection uniformity — no two
points share the same stratum in any coordinate dimension.
"""
function lhs_sample(n_vars::Int, n_pts::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    pts = zeros(n_pts, n_vars)
    for j in 1:n_vars
        perm       = randperm(rng, n_pts)
        u          = rand(rng, n_pts)
        pts[:, j]  = (perm .- 1 .+ u) ./ n_pts   # stratum midpoint + jitter
    end
    return pts
end

# ---------------------------------------------------------------------------
# 7. Coded ↔ Natural variable helpers
# ---------------------------------------------------------------------------

"""Map coded [0,1]^d points to natural variable space defined by bounds (n×2)."""
function to_natural(pts_coded::Matrix{Float64}, bounds::Matrix{Float64})
    pts_nat = similar(pts_coded)
    for j in 1:size(pts_coded, 2)
        lo, hi = bounds[j, 1], bounds[j, 2]
        pts_nat[:, j] = lo .+ pts_coded[:, j] .* (hi - lo)
    end
    return pts_nat
end

"""Map natural variable vector back to coded [0,1] given bounds."""
function to_coded(x_nat::Vector{Float64}, bounds::Matrix{Float64})
    return [(x_nat[j] - bounds[j,1]) / (bounds[j,2] - bounds[j,1])
            for j in 1:length(x_nat)]
end

# ---------------------------------------------------------------------------
# 8. Parallel Black-Box Evaluation
# ---------------------------------------------------------------------------

"""
    evaluate_parallel(fn, pts_nat) -> Vector

Evaluate `fn` at each row of `pts_nat` using Julia's multi-threading.
Safe for any thread-safe BB function. Falls back to `-Inf` on error.

Set `JULIA_NUM_THREADS` environment variable before starting Julia
(e.g. `JULIA_NUM_THREADS=8 julia`) to use multiple threads.
"""
function evaluate_parallel(fn, pts_nat::Matrix{Float64})
    n        = size(pts_nat, 1)
    results  = Vector{Float64}(undef, n)
    n_errors = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:n
        results[i] = try
            fn(pts_nat[i, :])
        catch e
            Threads.atomic_add!(n_errors, 1)
            -Inf
        end
    end
    n_errors[] > 0 && @warn "evaluate_parallel: $(n_errors[]) / $n evaluations failed (returned -Inf)"
    return results
end

# ---------------------------------------------------------------------------
# 9. Trust-Region Bounds Update
# ---------------------------------------------------------------------------

"""
    trust_region_update(x_opt_coded, bounds; K, expand_factor, oscillation_tol)
        -> Matrix (new bounds)

Update design variable bounds using a trust-region strategy:
- Contract by factor K around the optimum (tighter than BLISS-RS's fixed 20%).
- Expand by `expand_factor` if the optimum hits the same boundary twice (oscillation
  detection requires `prev_opt` in natural space from the previous outer iteration).
- Normal contraction clamps to current physical bounds; the expansion path allows
  bounds to grow beyond the original lo/hi to escape wall-hugging optima.

Improvement over BLISS-RS: the contraction factor K is applied in the coded space
after re-centering on x_opt, so the new interval is always symmetric around the best
found point (within physical limits). This prevents drift toward boundaries.
"""
function trust_region_update(x_opt::Vector{Float64}, bounds::Matrix{Float64};
                              K::Float64=0.7,
                              expand_factor::Float64=1.3,
                              prev_opt::Union{Nothing,Vector{Float64}}=nothing)
    n = length(x_opt)
    new_bounds = copy(bounds)

    for i in 1:n
        lo, hi = bounds[i, 1], bounds[i, 2]
        width  = hi - lo
        half   = K * width / 2.0

        new_lo = max(lo, x_opt[i] - half)
        new_hi = min(hi, x_opt[i] + half)

        # Expand if optimum is at boundary (oscillation / wall detection)
        if prev_opt !== nothing
            if abs(x_opt[i] - lo) < 1e-4 * width && abs(prev_opt[i] - lo) < 1e-4 * width
                new_lo = max(lo - expand_factor * half, lo - width * expand_factor)
                new_hi = min(hi, new_lo + expand_factor * width)
            end
        end

        # Ensure positive width
        if new_hi - new_lo < 1e-6 * width
            new_lo = max(lo, x_opt[i] - half)
            new_hi = min(hi, x_opt[i] + half)
        end

        new_bounds[i, 1] = new_lo
        new_bounds[i, 2] = new_hi
    end

    return new_bounds
end

# ---------------------------------------------------------------------------
# 10. BLISS-2026 Main Loop
# ---------------------------------------------------------------------------

"""
    bliss_2026(bb_fn, n_vars, bounds; kwargs...) -> NamedTuple

**BLISS-2026** main optimisation loop.

Arguments:
- `bb_fn(Z)`: scalar-valued black-box function (to maximise), Z a natural variable vector.
- `n_vars`: number of system design variables.
- `bounds`: n×2 matrix of [lo, hi] physical bounds.

Keyword arguments:
- `n_lhs`:       LHS points per outer iteration (initial DOE size).
- `n_bo`:        Bayesian optimisation steps per outer iteration.
- `n_candidates`:random candidates evaluated under EI at each BO step.
- `K`:           trust-region contraction factor (0 < K < 1, default 0.7).
- `n_outer`:     number of outer BLISS iterations.
- `σ_n`:         GP noise level (1e-3 works for normalised targets).
- `ξ`:           EI exploration bonus.
- `seed`:        random seed for reproducibility.
- `verbose`:     print iteration summary.

Returns: `(Z_history, obj_history, gp_models, bounds_history)`
"""
function bliss_2026(bb_fn, n_vars::Int, bounds::Matrix{Float64};
                     n_lhs::Int       = max(2 * n_vars + 3, 15),
                     n_bo::Int        = 15,
                     n_candidates::Int= 1000,
                     K::Float64       = 0.7,
                     n_outer::Int     = 8,
                     σ_n::Float64     = 1e-3,
                     ξ::Float64       = 0.01,
                     seed::Int        = 42,
                     verbose::Bool    = true)

    rng            = MersenneTwister(seed)
    current_bounds = copy(bounds)
    prev_Z_opt     = nothing   # natural-space optimum from previous outer iter

    Z_history      = Vector{Vector{Float64}}()
    obj_history    = Float64[]
    bounds_history = Vector{Matrix{Float64}}()
    gp_out         = Any[]

    if verbose
        @printf("\n  %-6s  %-10s  %-10s  %-8s  %-8s\n",
                "Iter", "Obj*", "ΔObj", "n_train", "GP ℓ")
        println("  " * "─"^50)
    end

    for outer in 1:n_outer

        # ── 1. Latin Hypercube design in current bounds ──────────────────────
        pts_coded = lhs_sample(n_vars, n_lhs; seed=seed + outer * 97)
        pts_nat   = to_natural(pts_coded, current_bounds)

        # ── 2. Parallel BB evaluation ─────────────────────────────────────────
        y_vals = evaluate_parallel(bb_fn, pts_nat)

        # ── 3. Fit GP with hyperparameter optimisation ────────────────────────
        gp, y_mean, y_std = fit_gp_auto(pts_coded, y_vals; σ_n=σ_n)

        # ── 4. Bayesian optimisation inner loop (EI) ──────────────────────────
        X_train = copy(pts_coded)
        y_train = copy(y_vals)

        for bo in 1:n_bo
            # Normalise current training set
            y_norm   = (y_train .- mean(y_train)) ./ (std(y_train) + 1e-8)
            gp_inner = fit_gp(X_train, y_norm, gp.kernel; σ_n=σ_n)

            f_best_norm = maximum(y_norm)

            # Sample candidates uniformly in coded space
            cands  = rand(rng, n_candidates, n_vars)
            μ_c, σ²_c = predict_gp(gp_inner, cands)
            ei_c   = expected_improvement(μ_c, σ²_c, f_best_norm; ξ=ξ)

            i_best = argmax(ei_c)
            x_next = cands[i_best, :]
            Z_next = to_natural(reshape(x_next, 1, :), current_bounds)[1, :]

            y_next = try bb_fn(Z_next) catch; -Inf end

            X_train = vcat(X_train, x_next')
            y_train = vcat(y_train, y_next)
        end

        # ── 5. Best point from full training set ─────────────────────────────
        i_star       = argmax(y_train)
        x_opt_coded  = X_train[i_star, :]
        Z_opt        = to_natural(reshape(x_opt_coded, 1, :), current_bounds)[1, :]
        obj_opt      = y_train[i_star]

        push!(Z_history,      Z_opt)
        push!(obj_history,    obj_opt)
        push!(bounds_history, copy(current_bounds))
        push!(gp_out,         (gp=gp, y_mean=y_mean, y_std=y_std,
                                X_train=X_train, y_train=y_train))

        # ── 6. Trust-region bounds update ─────────────────────────────────────
        current_bounds = trust_region_update(Z_opt, current_bounds; K=K, prev_opt=prev_Z_opt)
        prev_Z_opt = Z_opt

        # ── 7. Verbose summary ────────────────────────────────────────────────
        if verbose
            δobj = outer > 1 ? obj_opt - obj_history[end-1] : 0.0
            ℓ_val = gp.kernel.ℓ
            @printf("  %-6d  %-10.2f  %-+10.2f  %-8d  %-8.4f\n",
                    outer, obj_opt, δobj, size(X_train, 1), ℓ_val)
        end
    end

    return (Z_history     = Z_history,
            obj_history   = obj_history,
            bounds_history= bounds_history,
            gp_models     = gp_out)
end

# ---------------------------------------------------------------------------
# 11. Comparison Utilities
# ---------------------------------------------------------------------------

"""
    gp_coverage_test(gp, y_mean, y_std, X_test, y_test) -> NamedTuple

Assess GP predictive calibration:
- Coverage: fraction of test points inside the 95% predictive interval.
- RMSE: root mean squared error of posterior mean.
- MNLL: mean negative log-likelihood under the posterior.
"""
function gp_coverage_test(gp_norm::GPModel, y_mean::Float64, y_std::Float64,
                            X_test::Matrix{Float64}, y_test::Vector{Float64})
    μ, σ² = predict_gp_scaled(gp_norm, y_mean, y_std, X_test)
    σ     = sqrt.(max.(σ², 1e-12))
    z     = (y_test .- μ) ./ σ
    coverage = mean(abs.(z) .< 1.96)
    rmse     = sqrt(mean((y_test .- μ).^2))
    mnll     = mean(0.5 * z.^2 .+ log.(σ) .+ 0.5 * log(2π))
    return (coverage=coverage, rmse=rmse, mnll=mnll)
end

"""
    compare_rs_vs_gp(bb_fn, n_vars, bounds; n_pts, n_test, seed) -> NamedTuple

Side-by-side comparison of quadratic RS (BLISS-RS) and GP (BLISS-2026) accuracy
on the same training set, evaluated on held-out test points.

Requires `fit_rs` and `predict_rs` from `bliss_rs.jl` to be loaded before calling.
"""
function compare_rs_vs_gp(bb_fn, n_vars::Int, bounds::Matrix{Float64};
                            n_pts::Int=40, n_test::Int=200, seed::Int=42)
    rng = MersenneTwister(seed)

    # Training data (LHS)
    X_train_c = lhs_sample(n_vars, n_pts; seed=seed)
    X_train_n = to_natural(X_train_c, bounds)
    y_train   = evaluate_parallel(bb_fn, X_train_n)

    # Test data (independent random)
    X_test_c  = rand(rng, n_test, n_vars)
    X_test_n  = to_natural(X_test_c, bounds)
    y_test    = evaluate_parallel(bb_fn, X_test_n)

    # ── Quadratic RS (from bliss_rs.jl) ──────────────────────────────────────
    b_rs      = fit_rs(X_train_c, y_train)
    y_rs_pred = predict_rs(X_test_c, b_rs)
    rmse_rs   = sqrt(mean((y_test .- y_rs_pred).^2))
    mae_rs    = mean(abs.(y_test .- y_rs_pred))

    # ── GP (BLISS-2026) ───────────────────────────────────────────────────────
    gp_norm, y_mean, y_std = fit_gp_auto(X_train_c, y_train)
    μ_gp, σ²_gp = predict_gp_scaled(gp_norm, y_mean, y_std, X_test_c)
    rmse_gp   = sqrt(mean((y_test .- μ_gp).^2))
    mae_gp    = mean(abs.(y_test .- μ_gp))
    coverage  = mean(abs.((y_test .- μ_gp) ./ sqrt.(max.(σ²_gp, 1e-12))) .< 1.96)

    return (rmse_rs=rmse_rs, mae_rs=mae_rs,
            rmse_gp=rmse_gp, mae_gp=mae_gp, coverage_gp=coverage,
            y_test=y_test, y_rs=y_rs_pred, y_gp=μ_gp, σ_gp=sqrt.(σ²_gp))
end
