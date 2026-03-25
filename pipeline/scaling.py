import numpy as np

# ================================================================
# SECCIÓN 1: FUNCIÓN PRINCIPAL
# ================================================================

def compute_params(Re_dp, dp_m, phi, Lx_m, Ly_m, rho_phy, mu_phy, tau, Nx):
    """
    Calcula todos los parámetros derivados para la simulación LBM
    y para las ecuaciones analíticas de comparación.

    Entradas:
        Re_dp   : Reynolds basado en diámetro de partícula
        dp_m    : diámetro de partícula [m]
        phi     : porosidad (0 < phi < 1)
        Lx_m    : longitud del canal [m]
        Ly_m    : altura del canal [m]
        rho_phy : densidad del fluido [kg/m³]
        mu_phy  : viscosidad dinámica [Pa·s]
        tau     : parámetro de relajación LBM
        Nx      : celdas en x

    Retorna:
        p : diccionario con todos los parámetros derivados
    """

    # ── Propiedades del fluido ───────────────────────────────────
    nu_phy = mu_phy / rho_phy

    # ── Parámetros LBM ───────────────────────────────────────────
    nu_lbm = (tau - 0.5) / 3.0

    # ── Grilla ───────────────────────────────────────────────────
    c_l      = Lx_m / Nx
    Ny       = max(1, round(Ly_m / c_l))
    dp_cells = dp_m / c_l

    # ── Escala temporal ──────────────────────────────────────────
    c_t = c_l**2 * (nu_lbm / nu_phy)

    # ── Velocidades en lattice ───────────────────────────────────
    u_inlet_lbm = Re_dp * nu_lbm / dp_cells
    u_darcy_lbm = u_inlet_lbm * phi

    # ── Velocidades físicas ──────────────────────────────────────
    u_inlet_phy = u_inlet_lbm * (c_l / c_t)
    u_darcy_phy = u_darcy_lbm * (c_l / c_t)

    # ── Números adimensionales ───────────────────────────────────
    Ma   = u_inlet_lbm / (1.0 / np.sqrt(3.0))
    Re_L = u_inlet_phy * Ly_m / nu_phy

    # ── Kozeny-Carman ────────────────────────────────────────────
    phi_s  = min(max(phi, 0.001), 0.999)
    K_phy  = (phi_s**3 * dp_m**2) / (150.0 * (1.0 - phi_s)**2)
    K_lbm  = K_phy / c_l**2
    F_eps  = 1.75 / np.sqrt(150.0 * phi_s**3)

    # ── Gradientes de presión analíticos ────────────────────────
    dP_L_darcy = (mu_phy / K_phy) * u_darcy_phy
    dP_L_ergun = dP_L_darcy + \
                 (F_eps * rho_phy * u_darcy_phy**2) / np.sqrt(K_phy)

    # ── Tiempo de simulación ─────────────────────────────────────
    t_diff = int(Ny*10 / nu_lbm)
    t_max  = t_diff * 5

    # ── Empacar todo en un diccionario ───────────────────────────
    p = {
        # inputs
        "Re_dp"       : Re_dp,
        "dp_m"        : dp_m,
        "phi"         : phi,
        "Lx_m"        : Lx_m,
        "Ly_m"        : Ly_m,
        "rho_phy"     : rho_phy,
        "mu_phy"      : mu_phy,
        "nu_phy"      : nu_phy,
        # grilla
        "Nx"          : Nx,
        "Ny"          : Ny,
        "c_l"         : c_l,
        "c_t"         : c_t,
        "dp_cells"    : dp_cells,
        # LBM
        "tau"         : tau,
        "nu_lbm"      : nu_lbm,
        "u_inlet_lbm" : u_inlet_lbm,
        "u_darcy_lbm" : u_darcy_lbm,
        # físico
        "u_inlet_phy" : u_inlet_phy,
        "u_darcy_phy" : u_darcy_phy,
        "Ma"          : Ma,
        "Re_dp_check" : u_inlet_phy * dp_m / nu_phy,
        "Re_Ly"       : Re_L,
        # medio poroso
        "K_phy"       : K_phy,
        "K_lbm"       : K_lbm,
        "F_eps"       : F_eps,
        # analítico
        "dP_L_darcy"  : dP_L_darcy,
        "dP_L_ergun"  : dP_L_ergun,
        # tiempo
        "t_diff"      : t_diff,
        "t_max"       : t_max,
    }

    return p


# ================================================================
# SECCIÓN 2: VALIDACIÓN
# ================================================================

def is_stable(p):
    """
    Verifica estabilidad numérica y compatibilidad física.
    Retorna (bool, lista de errores, lista de advertencias).
    """
    errors   = []
    warnings = []

    if p["Ma"] >= 0.1:
        errors.append(
            f"❌ Ma = {p['Ma']:.4f} ≥ 0.1 — reducir Re_dp o aumentar Nx"
        )
    elif p["Ma"] >= 0.05:
        warnings.append(
            f"⚠️  Ma = {p['Ma']:.4f} — cerca del límite"
        )

    if p["tau"] <= 0.5:
        errors.append(f"❌ tau = {p['tau']} ≤ 0.5 — inestable")

    if p["dp_cells"] >= p["Ny"] / 2:
        errors.append(
            f"❌ dp = {p['dp_cells']:.1f} celdas ≥ Ny/2 = {p['Ny']/2:.1f}"
            f" — partícula más grande que el canal"
        )

    if p["dp_cells"] < 5:
        warnings.append(
            f"⚠️  dp = {p['dp_cells']:.1f} celdas — resolución baja, "
            f"recomendado ≥ 5"
        )

    if not (0.35 <= p["phi"] <= 0.95):
        warnings.append(
            f"⚠️  φ = {p['phi']} fuera del rango válido de "
            f"Kozeny-Carman (0.35 - 0.95)"
        )

    stable = len(errors) == 0
    return stable, errors, warnings


# ================================================================
# SECCIÓN 3: RESUMEN
# ================================================================

def print_summary(p, label=""):
    stable, errors, warnings = is_stable(p)
    status = "✔  ESTABLE" if stable else "❌  INESTABLE"

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         PARÁMETROS DE SIMULACIÓN LBM D2Q9            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"  Estado            : {status}")
    print(f"  Caso              : {label or 'sin etiqueta'}")
    print(f"  Re_dp             : {p['Re_dp']}")
    print(f"  dp                : {p['dp_m']*100:.2f} cm"
          f"  →  {p['dp_cells']:.1f} celdas")
    print(f"  Porosidad φ       : {p['phi']:.4f}")
    print(f"  Canal             : {p['Lx_m']:.4f} m × {p['Ly_m']:.4f} m")
    print(f"  Grilla            : {p['Nx']} × {p['Ny']} celdas")
    print(f"  Δx                : {p['c_l']:.4e} m/celda")
    print(f"  Δt                : {p['c_t']:.4e} s/paso")
    print(f"  tau               : {p['tau']:.2f}")
    print(f"  nu_lbm            : {p['nu_lbm']:.6f}")
    print(f"  u_inlet_lbm       : {p['u_inlet_lbm']:.6f} [lu/ts]")
    print(f"  u_darcy_lbm       : {p['u_darcy_lbm']:.6f} [lu/ts]")
    print(f"  Ma                : {p['Ma']:.4f}")
    print(f"  u_inlet_phy       : {p['u_inlet_phy']:.4e} m/s")
    print(f"  u_darcy_phy       : {p['u_darcy_phy']:.4e} m/s")
    print(f"  Re_dp resultante  : {p['Re_dp_check']:.4f}")
    print(f"  Re (L=Ly)         : {p['Re_Ly']:.4f}")
    print(f"  K_phy (K-C)       : {p['K_phy']:.4e} m²")
    print(f"  K_lbm             : {p['K_lbm']:.4e}")
    print(f"  F_eps (Ergun)     : {p['F_eps']:.4f}")
    print(f"  ΔP/L Darcy        : {p['dP_L_darcy']:.4e} Pa/m")
    print(f"  ΔP/L Ergun        : {p['dP_L_ergun']:.4e} Pa/m")
    print(f"  t_diff            : {p['t_diff']} pasos")
    print(f"  t_max             : {p['t_max']} pasos")

    print("╠══════════════════════════════════════════════════════╣")
    for w in warnings:
        print(f"  {w}")
    for e in errors:
        print(f"  {e}")
    print("╚══════════════════════════════════════════════════════╝\n")


# ================================================================
# VERIFICACIÓN RÁPIDA
# ================================================================

if __name__ == "__main__":
    p = compute_params(
        Re_dp   = 1.0,
        dp_m    = 0.005,
        phi     = 0.80,
        Lx_m    = 0.09,
        Ly_m    = 0.03,
        rho_phy = 1.204,
        mu_phy  = 1.825e-4,
        tau     = 0.8,
        Nx      = 600
    )
    print_summary(p, label="prueba_scaling")