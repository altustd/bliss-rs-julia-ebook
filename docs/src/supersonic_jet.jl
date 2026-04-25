# =============================================================================
# supersonic_jet.jl — Black-box analyses for the BLISS supersonic business jet
# Reproduced from Altus & Sobieski (2002), NASA/CR-2002-211652
# =============================================================================

# ---------------------------------------------------------------------------
# System variable layout (matches Altus 2002, Table 1)
# Z = [t_c, ARw, Λw, Sw, SHT, ARHT, λ]
#   t_c  : wing thickness-to-chord ratio  (–)
#   ARw  : wing aspect ratio              (–)
#   Λw   : wing sweep angle              (deg)
#   Sw   : wing surface area             (ft²)
#   SHT  : horizontal tail surface area  (ft²)
#   ARHT : tail aspect ratio             (–)
#   λ    : wing taper ratio              (–)
# ---------------------------------------------------------------------------

const DEG2RAD = π / 180.0

# ---------------------------------------------------------------------------
# Atmospheric model (standard atmosphere, simplified)
# Returns (ρ, a) at altitude h [ft]
# ---------------------------------------------------------------------------
function std_atmosphere(h_ft::Real)
    h_m   = h_ft * 0.3048
    T     = max(216.65, 288.15 - 0.0065 * h_m)   # K
    P     = 101325.0 * (T / 288.15)^5.2561        # Pa
    ρ     = P / (287.058 * T)                     # kg/m³  → convert later
    ρ_slug = ρ * 0.00194032                        # slug/ft³
    a_fps  = sqrt(1.4 * 1716.49 * T * 1.8)        # ft/s  (T in °R = T_K * 1.8)
    return ρ_slug, a_fps
end

# ---------------------------------------------------------------------------
# Structures Black Box
# Inputs: Z-variables + coupling Y (L from aero, WE from power)
# Outputs: structural weight Ws, "twist" Ω (reduction in effective lift area, ft²)
# ---------------------------------------------------------------------------
"""
    structures_bb(Z, L, WE) -> (Ws, Ω)

Simplified structures analysis.  Uses an empirical wingbox weight formula
and a linear aeroelastic twist model.

References: ELAPS model, Giles (1986, 1989); Altus (2002) §2.2
"""
function structures_bb(Z::AbstractVector, L::Real, WE::Real)
    t_c, ARw, Λw, Sw, _, _, λ = Z
    Λ_rad = Λw * DEG2RAD

    # Empirical structural weight (Raymer 1992, Ch. 15 simplified)
    # W_wing = 0.0051 * (W_dg * n_z)^0.557 * Sw^0.649 * ARw^0.5 /
    #          (t_c^0.4 * (1+λ)^0.1 * cos(Λ)^1.0 * (Sw_cs)^0.1)
    # We use a surrogate calibrated to BLISS-98 range:
    Ws = 2.0 * (Sw^0.65) * (ARw^0.5) / (t_c^0.4 * cos(Λ_rad) * (1 + λ)^0.2)

    # Aeroelastic twist: reduction in effective lift area (ft²)
    # Larger aspect ratio → more twist → more loss
    bending_stiffness = t_c * Sw / (ARw + 1e-6)
    Ω = 0.01 * ARw^1.2 * L / (bending_stiffness * 1e4 + 1.0)
    return Ws, Ω
end

# ---------------------------------------------------------------------------
# Aerodynamics Black Box
# Inputs: Z + coupling Y (WT, Ω, ESF from other BBs)
# Outputs: Lift L, Drag D, L/D
# ---------------------------------------------------------------------------
"""
    aero_bb(Z, h, M, WT, Ω, ESF) -> (L, D, LD)

Simplified drag-polar aerodynamics model (AWAVE surrogate).
Wave drag from Harris (1964); induced drag from lifting-line theory.

References: Altus (2002) §2.2; Harris (1964)
"""
function aero_bb(Z::AbstractVector, h::Real, M::Real,
                 WT::Real, Ω::Real, ESF::Real)
    t_c, ARw, Λw, Sw, SHT, ARHT, λ = Z
    Λ_rad = Λw * DEG2RAD
    ρ, a  = std_atmosphere(h)
    q     = 0.5 * ρ * (M * a)^2           # dynamic pressure (lbf/ft²)

    # Cruise: L = W_total
    L  = WT
    CL = L / (q * Sw + 1e-6)

    # Oswald efficiency
    e  = 0.85 * exp(-0.045 * ARw^0.68 * cos(Λ_rad)^0.15)

    # Drag polar components
    CDmin  = 0.012 + 0.0025 * M^2 + 0.0018 * t_c      # friction + form
    CDi    = CL^2 / (π * ARw * e)                      # induced
    CDwave = 0.0015 * max(0, M - 0.9)^2 * (1 + 4*(t_c - 0.05)^2)  # wave

    CD = CDmin + CDi + CDwave
    D  = CD * q * Sw
    LD = (D > 0) ? L / D : 0.0
    return L, D, LD
end

# ---------------------------------------------------------------------------
# Power Black Box
# Inputs: Z (h, M) + coupling D
# Outputs: SFC, ESF, WE
# ---------------------------------------------------------------------------
"""
    power_bb(h, M, D, T_throttle=1.0) -> (SFC, ESF, WE)

Engine deck approximation.  Calibrated to a generic high-bypass turbofan
at supersonic cruise conditions.  ESF = engine scale factor (–).

References: Altus (2002) §2.2
"""
function power_bb(h::Real, M::Real, D::Real; T_throttle::Real=1.0)
    ρ, _ = std_atmosphere(h)

    # SFC model (lb_fuel / (lbf_thrust · hr))
    SFC = (0.45 + 0.54 * M) * (ρ / 0.00237)^(-0.07)

    # Thrust available per engine (baseline twin, lbf)
    F_max_eng = 8000.0 * (ρ / 0.00237)^0.8 * (1.0 - 0.3 * max(M - 1.0, 0))

    # Engine scale factor to match required thrust = D
    F_needed = D / T_throttle
    ESF      = F_needed / (2 * F_max_eng + 1e-6)
    ESF      = clamp(ESF, 0.5, 2.5)

    # Engine weight (empirical)
    WE = 1200.0 * ESF^0.9   # lbf

    return SFC, ESF, WE
end

# ---------------------------------------------------------------------------
# Performance (Range) Black Box — Breguet Range Equation
# Inputs: coupling variables WT, WF, L/D, SFC; system h, M
# Output: Range (nautical miles)
# ---------------------------------------------------------------------------
"""
    performance_bb(WT, WF, LD, SFC, h, M) -> Range [NM]

Breguet range equation for cruise flight:
  R = (a·M / SFC) · (L/D) · ln(W_i / W_f)

References: Raymer (1992) Ch. 17; Altus (2002) §2.2
"""
function performance_bb(WT::Real, WF::Real, LD::Real, SFC::Real,
                         h::Real, M::Real)
    _, a = std_atmosphere(h)
    a_kts = a / 1.68781          # ft/s → knots
    Wi    = WT                   # initial cruise weight
    Wf    = WT - WF              # final cruise weight (fuel burned)
    if Wf <= 0 || LD <= 0 || SFC <= 0
        return 0.0
    end
    SFC_hr = SFC                 # already in lb/(lbf·hr)
    Range_hr = (a_kts * M / SFC_hr) * LD * log(Wi / Wf)
    return Range_hr              # nautical miles
end

# ---------------------------------------------------------------------------
# Full supersonic BJ system analysis (one BLISS analysis pass)
# ---------------------------------------------------------------------------
"""
    ssbj_analysis(Z, h, M, WF_init) -> NamedTuple

Run one coupled system analysis of the supersonic business jet.
Starting from initial fuel weight guess `WF_init`, iterates the
aero-structures-power loop to convergence.
"""
function ssbj_analysis(Z::AbstractVector, h::Real, M::Real;
                        WF_init::Real=15_000.0, WT_empty::Real=30_000.0,
                        tol::Float64=1e-3, max_iter::Int=50)
    WF  = WF_init
    WT  = WT_empty + WF
    Ω   = 0.0
    ESF = 1.0

    for _ in 1:max_iter
        # Aero analysis
        L, D, LD = aero_bb(Z, h, M, WT, Ω, ESF)
        # Power analysis
        SFC, ESF_new, WE = power_bb(h, M, D)
        # Structures analysis
        Ws, Ω_new = structures_bb(Z, L, WE)

        # Update total weight
        WT_new  = Ws + WE + WF + 10_000.0   # add payload/systems fixed weight
        WF_new  = SFC * D / (M * 661.5) * 0.85 * WF / (WT + 1e-6) * WF  # simplified

        if abs(WT_new - WT) / (WT + 1e-6) < tol
            WT, WF, ESF, Ω = WT_new, WF_new, ESF_new, Ω_new
            break
        end
        WT, WF, ESF, Ω = WT_new, WF_new, ESF_new, Ω_new
    end

    L, D, LD = aero_bb(Z, h, M, WT, Ω, ESF)
    SFC, _, WE = power_bb(h, M, D)
    Range = performance_bb(WT, WF, LD, SFC, h, M)

    return (WT=WT, WF=WF, ESF=ESF, Ω=Ω, L=L, D=D, LD=LD, SFC=SFC, Range=Range)
end
