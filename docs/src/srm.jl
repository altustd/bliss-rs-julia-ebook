# =============================================================================
# srm.jl — Solid Propellant Rocket Motor (SRM) Black-Box analyses
# Application of BLISS-RS to SRM design (Chapter 10)
# =============================================================================
#
# Design variable decomposition:
#
#   System variables Z (shared across BBs):
#     L_D       : grain length-to-diameter ratio     (–)
#     Pc_target : target chamber pressure            (psia)
#     epsilon   : nozzle area expansion ratio        (–)
#     Phi_g     : volumetric loading fraction        (–)
#
#   BB 1 — Internal Ballistics (Propulsion):
#     Local X: [AP_pct, Al_pct, r_a, r_n]
#       AP_pct : ammonium perchlorate % by mass      (%)
#       Al_pct : aluminum % by mass                  (%)
#       r_a    : burn rate pre-exponential (a)       (in/s · psia^-n)
#       r_n    : burn rate exponent (n)              (–)
#     Outputs (coupling Y): Pc, Fthrust, Isp, mdot, t_burn
#
#   BB 2 — Structural Analysis:
#     Local X: [t_case, sigma_allow]
#       t_case     : case wall thickness             (in)
#       sigma_allow: allowable stress               (ksi)
#     Outputs: W_case, FS_burst
#
#   BB 3 — Nozzle / Performance:
#     Local X: [half_angle, mat_rho]
#       half_angle : nozzle divergence half-angle    (deg)
#       mat_rho    : nozzle material density        (lb/in³)
#     Outputs: CF, W_nozzle, Isp_vac
#
#   BB 4 — Mission Performance:
#     Coupling Y: (total_impulse, W_total)
#     Output: delta_V (ft/s), mass_ratio
#
# =============================================================================

using LinearAlgebra, Statistics

const G0_FPS  = 32.174    # ft/s²
const G0_IPS  = 386.09    # in/s²
const PSI2LBF = 1.0       # 1 psia * 1 in² = 1 lbf

# ---------------------------------------------------------------------------
# Ideal gas nozzle thermodynamics
# ---------------------------------------------------------------------------
"""
    thrust_coefficient(gamma, Pc, Pe, Pa, epsilon) -> CF

Compute the nozzle thrust coefficient CF for ideal isentropic flow.
  CF = sqrt(2γ²/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · [1-(Pe/Pc)^((γ-1)/γ)])
     + (Pe - Pa)/Pc · ε
"""
function thrust_coefficient(gamma::Real, Pc::Real, Pe::Real, Pa::Real,
                             epsilon::Real)
    g = gamma
    arg = 2*g^2/(g-1) *
          (2/(g+1))^((g+1)/(g-1)) *
          max(0.0, 1 - (min(Pe, Pc*0.999) / Pc)^((g-1)/g))
    CF_thrust = sqrt(arg)
    CF_press  = (Pe - Pa) / Pc * epsilon
    return CF_thrust + CF_press
end

"""
    exit_pressure(Pc, gamma, epsilon) -> Pe (psia)

Isentropic area-Mach relation solved for exit pressure given expansion ratio ε.
Uses Newton iteration.
"""
function exit_pressure(Pc::Real, gamma::Real, epsilon::Real)
    g = gamma
    epsilon = max(epsilon, 1.001)   # expansion ratio must be > 1
    A_ratio(Me) = (1/Me) * ((2/(g+1)) * (1 + (g-1)/2 * Me^2))^((g+1)/(2*(g-1)))
    Me = max(1.5, sqrt(epsilon))    # better initial guess
    for _ in 1:100
        ar  = A_ratio(Me)
        dar = (A_ratio(Me * 1.0005) - ar) / (Me * 0.0005)
        δ   = (ar - epsilon) / (dar + 1e-12)
        Me  = max(1.001, Me - clamp(δ, -0.5, 0.5))
    end
    Pe = Pc * (1 + (g-1)/2 * Me^2)^(-g/(g-1))
    return max(Pe, 1e-3)   # prevent negative exit pressure
end

"""
    characteristic_velocity(gamma, Tc, MW) -> c_star (ft/s)

Characteristic velocity c* for ideal combustion.
  c* = sqrt(γ R Tc / γ) / Γ(γ)   where Γ(γ) = γ · sqrt((2/(γ+1))^((γ+1)/(γ-1)))
"""
function characteristic_velocity(gamma::Real, Tc::Real, MW::Real)
    R_univ = 1545.0         # ft·lbf/(lbmol·°R)
    R_gas  = R_univ / MW    # ft·lbf/(lbm·°R)
    Γ = gamma * sqrt((2/(gamma+1))^((gamma+1)/(gamma-1)))
    c_star = sqrt(gamma * R_gas * Tc) / Γ
    return c_star            # ft/s
end

# ---------------------------------------------------------------------------
# Propellant thermochemical properties (AP/Al/HTPB family)
# Simple surrogate based on NASA-CEA-style curve fits
# ---------------------------------------------------------------------------
"""
    propellant_properties(AP_pct, Al_pct) -> (Tc, gamma, MW, rho_prop)

Return approximate flame temperature (°R), γ, molecular weight (lbm/lbmol),
and propellant bulk density (lb/in³) for an AP/Al/HTPB composite.
Calibrated to published data for typical APCP compositions.
"""
function propellant_properties(AP_pct::Real, Al_pct::Real)
    # Flame temperature (°R): peaks near 18–20% Al, high AP loading
    Tc = 5400.0 + 40 * (AP_pct - 68) + 80 * (Al_pct - 18) -
         2.0 * (AP_pct - 68)^2 - 3.0 * (Al_pct - 18)^2

    # Specific heat ratio γ: decreases with Al (two-phase effects)
    gamma = 1.225 - 0.003 * Al_pct

    # Effective MW of combustion products (lbm/lbmol)
    MW = 25.0 + 0.2 * Al_pct - 0.05 * (AP_pct - 68)

    # Propellant bulk density (lb/in³)
    # AP ~0.069, Al ~0.097, HTPB binder ~0.033
    HTPB_pct = 100.0 - AP_pct - Al_pct
    rho_prop = (AP_pct * 0.069 + Al_pct * 0.097 + HTPB_pct * 0.033) / 100.0

    return Tc, gamma, MW, rho_prop
end

# ---------------------------------------------------------------------------
# BB 1 — Internal Ballistics
# ---------------------------------------------------------------------------
"""
    ballistics_bb(Z, X_prop) -> NamedTuple

Internal ballistics black box.

Z = [L_D, Pc_target, epsilon, Phi_g]   (system variables)
X = [AP_pct, Al_pct, r_a, r_n]         (local variables)

Key equations:
  r = a · Pc^n          (Saint-Robert's burn rate law)
  Pc = (a · ρp · Ab · c* / At)^(1/(1-n))   (chamber pressure balance)
  F  = CF · At · Pc
  Isp = F / (ṁ · g0)

Reference: Sutton & Biblarz (2010) Ch. 14–15
"""
function ballistics_bb(Z::AbstractVector, X::AbstractVector)
    L_D, Pc_tgt, epsilon, Phi_g = Z
    AP_pct, Al_pct, r_a, r_n   = X

    # Clamp all inputs to physically meaningful ranges
    L_D    = max(1.0, L_D)
    Pc_tgt = max(100.0, Pc_tgt)
    epsilon = max(1.5, epsilon)
    Phi_g  = clamp(Phi_g, 0.30, 0.95)          # port must exist; max packing ~0.95
    AP_pct = clamp(AP_pct, 50.0, 80.0)
    Al_pct = clamp(Al_pct, 5.0, 25.0)
    r_a    = max(0.005, r_a)
    r_n    = clamp(r_n, 0.1, 0.7)

    # Motor geometry derived from Z
    D_out    = 6.0                    # outer diameter (in), fixed reference
    R_out    = D_out / 2.0
    L_grain  = L_D * D_out           # grain length (in)

    # Propellant thermochemical properties
    Tc, gamma, MW, rho_p = propellant_properties(AP_pct, Al_pct)

    # Characteristic velocity (ft/s → in/s for consistency)
    c_star_fps = characteristic_velocity(gamma, Tc, MW)
    c_star     = c_star_fps * 12.0   # in/s

    # Grain geometry (BATES grain: hollow cylinder)
    R_port   = R_out * sqrt(1.0 - Phi_g)         # port radius (in)
    A_b      = 2π * R_port * L_grain             # initial burn surface (in²)
    V_prop   = π * (R_out^2 - R_port^2) * L_grain  # propellant volume (in³)
    W_prop   = rho_p * V_prop                    # propellant weight (lbf, density in lb/in³)

    # Throat area from target Pc: At = (a·ρp·Ab·c*) / Pc^n / Pc = …
    # Pc = (ρp·Ab·c*·a / At)^(1/(1-n))  => At = ρp·Ab·c*·a / Pc^n / Pc
    Pc_tgt_psi = Pc_tgt                         # psia
    At = rho_p * A_b * c_star * r_a /
         (Pc_tgt_psi^(1 - r_n) + 1e-6)          # in²

    # Burn rate at target Pc
    r_dot = r_a * Pc_tgt_psi^r_n               # in/s

    # Mass flow rate  ṁ = ρp · Ab · r
    mdot = rho_p * A_b * r_dot                  # lb/s (lbm/s, using ρ in lb/in³ and in/s → lb/in²/s? No…)
    # Units: rho_p [lb/in³] * Ab [in²] * r [in/s] = lb/s  ✓ (lb = lbm here, weight)

    # Burn time
    web = R_out - R_port                        # web thickness (in)
    t_burn = web / r_dot                         # s

    # Exit pressure (isentropic expansion)
    Pa  = 14.696                                 # ambient (sea level, psia)
    Pe  = exit_pressure(Pc_tgt_psi, gamma, epsilon)

    # Thrust coefficient and thrust
    CF  = thrust_coefficient(gamma, Pc_tgt_psi, Pe, Pa, epsilon)
    F   = CF * At * Pc_tgt_psi                  # lbf

    # Specific impulse (s)
    Isp = F / (mdot * G0_IPS / 12.0)            # s  (g0 in in/s² → 32.174 ft/s² × 12 = 386.09 in/s²)
    Isp = F / (mdot * G0_FPS)                   # s  (consistent units: F[lbf], mdot[lb/s], g0[ft/s²])

    # Total impulse
    It = F * t_burn                              # lbf·s

    return (Pc=Pc_tgt_psi, F=F, Isp=Isp, mdot=mdot, t_burn=t_burn,
            It=It, At=At, A_b=A_b, W_prop=W_prop, r_dot=r_dot, CF=CF)
end

# ---------------------------------------------------------------------------
# BB 2 — Structural Analysis
# ---------------------------------------------------------------------------
"""
    structural_bb(Z, X_struct, Pc) -> NamedTuple

Thin-walled pressure vessel analysis for cylindrical composite motor case.

  σ_hoop  = Pc · R / t    (hoop stress, thin-walled Lamé)
  FS_burst = σ_allow / σ_hoop
  W_case   = ρ_case · π · D · t · L_grain

Local X = [t_case (in), sigma_allow (ksi)]
"""
function structural_bb(Z::AbstractVector, X::AbstractVector, Pc::Real)
    L_D, _, _, Phi_g = Z
    t_case, sigma_allow_ksi = X
    sigma_allow = sigma_allow_ksi * 1000.0       # psi

    D_out   = 6.0
    R_mid   = D_out / 2.0 - t_case / 2.0        # mean radius (in)
    L_grain = L_D * D_out                        # grain length (in)

    # Case material: filament-wound graphite/epoxy
    rho_case = 0.056    # lb/in³ (graphite/epoxy ~0.056, steel ~0.284)

    # Hoop stress (thin cylinder)
    sigma_hoop = Pc * R_mid / t_case             # psi

    # Factor of safety
    FS_burst = sigma_allow / (sigma_hoop + 1e-6)

    # Case weight (cylindrical shell + 2 dome closures ≈ 1.3× cylinder)
    A_case  = π * D_out * L_grain + 2 * π * (D_out/2)^2   # in²  (cylinder + flat caps)
    W_case  = rho_case * A_case * t_case * 1.3             # lbf

    return (sigma_hoop=sigma_hoop, FS_burst=FS_burst, W_case=W_case)
end

# ---------------------------------------------------------------------------
# BB 3 — Nozzle / Performance
# ---------------------------------------------------------------------------
"""
    nozzle_bb(Z, X_nozzle, Pc, At, CF) -> NamedTuple

Conical nozzle weight and vacuum Isp calculation.

Local X = [half_angle (deg), mat_rho (lb/in³)]
"""
function nozzle_bb(Z::AbstractVector, X::AbstractVector,
                   Pc::Real, At::Real, CF::Real)
    _, _, epsilon, _ = Z
    half_angle_deg, mat_rho = X

    # Nozzle throat and exit geometry
    Rt   = sqrt(At / π)                          # throat radius (in)
    Re   = Rt * sqrt(epsilon)                    # exit radius (in)
    ϕ    = half_angle_deg * π / 180.0            # half-angle (rad)

    # Conical nozzle length
    L_noz = (Re - Rt) / tan(ϕ)                  # in

    # Divergence efficiency factor λ for conical nozzle
    lambda = (1.0 + cos(ϕ)) / 2.0

    # Effective CF with divergence loss
    CF_eff = CF * lambda

    # Vacuum Isp (no ambient pressure subtraction)
    # CF_vac ≈ CF_eff + epsilon * Pe/Pc  — already included in CF
    # Isp_vac ≈ CF_eff * c_star / g0  — returned from ballistics, so scale
    Isp_vac_factor = CF_eff / CF              # relative gain vs. sea-level

    # Nozzle volume (conical frustum + throat region)
    V_noz   = π / 3.0 * L_noz * (Rt^2 + Rt*Re + Re^2)
    W_nozzle = mat_rho * V_noz * 1.5           # 1.5× for throat insert, mounts

    return (CF_eff=CF_eff, W_nozzle=W_nozzle, Isp_vac_factor=Isp_vac_factor,
            L_noz=L_noz)
end

# ---------------------------------------------------------------------------
# BB 4 — Mission Performance
# ---------------------------------------------------------------------------
"""
    mission_bb(W_total, W_prop, It, Isp) -> NamedTuple

Mission performance black box: delta-V from Tsiolkovsky rocket equation.
  ΔV = Isp · g0 · ln(W_total / W_burnout)

Returns ΔV (ft/s), propellant mass fraction (PMF), and range factor (RD).
"""
function mission_bb(W_total::Real, W_prop::Real, It::Real, Isp::Real)
    W_burnout = W_total - W_prop                 # lbf (weight at burnout)
    if W_burnout <= 0.0 || W_total <= 0.0
        return (delta_V=0.0, PMF=0.0, It=It)
    end
    mass_ratio = W_total / W_burnout
    delta_V    = Isp * G0_FPS * log(mass_ratio)  # ft/s
    PMF        = W_prop / W_total                # propellant mass fraction

    return (delta_V=delta_V, PMF=PMF, It=It)
end

# ---------------------------------------------------------------------------
# Full SRM analysis (one coupled pass)
# ---------------------------------------------------------------------------
"""
    srm_analysis(Z, X_prop, X_struct, X_nozzle) -> NamedTuple

Coupled SRM system analysis: chain all four black boxes.
"""
function srm_analysis(Z::AbstractVector, X_prop::AbstractVector,
                      X_struct::AbstractVector, X_nozzle::AbstractVector;
                      W_payload::Real=50.0)
    b = ballistics_bb(Z, X_prop)
    s = structural_bb(Z, X_struct, b.Pc)
    n = nozzle_bb(Z, X_nozzle, b.Pc, b.At, b.CF)

    W_total = b.W_prop + s.W_case + n.W_nozzle + W_payload   # lbf
    m = mission_bb(W_total, b.W_prop, b.It, b.Isp)

    return merge(b, s, n, m, (W_total=W_total,))
end
