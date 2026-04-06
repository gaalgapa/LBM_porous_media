# ================================================================
# postprocess.py
# Lee los resultados CSV de la simulación y calcula:
#   - Permeabilidad efectiva
#   - Caída de presión
#   - Error vs soluciones analíticas
#   - Estadísticas entre semillas
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# SECCIÓN 1: LECTURA DE RESULTADOS
# ================================================================

def load_metrics(run_dir):
    """
    Carga el CSV de métricas finales de una run.
    Retorna un DataFrame con una fila por semilla.

    Estructura esperada del CSV:
        seed, t_final, converged,
        U_darcy, rho_inlet, rho_outlet,
        dP_inlet, dP_outlet, dP_L
    """
    run_dir = Path(run_dir)
    fpath   = run_dir / "results" / "metrics.csv"

    if not fpath.exists():
        raise FileNotFoundError(f"No se encontró metrics.csv en {run_dir}")

    df = pd.read_csv(fpath)
    return df


def load_velocity_profile(run_dir, seed=0):
    """
    Carga el perfil de velocidad promediado en x para una semilla.
    Retorna arrays (y_cells, ux_mean).

    Estructura esperada del CSV:
        y, ux_mean, uy_mean, obstacle_fraction
    """
    run_dir = Path(run_dir)
    fpath   = run_dir / "results" / f"profile_seed{seed}.csv"

    if not fpath.exists():
        raise FileNotFoundError(
            f"No se encontró profile_seed{seed}.csv en {run_dir}"
        )

    df = pd.read_csv(fpath)
    return df


# ================================================================
# SECCIÓN 2: CÁLCULO DE MÉTRICAS
# ================================================================

def compute_K_effective(df_metrics, p):
    """
    Calcula la permeabilidad efectiva desde los resultados LBM
    usando la ley de Darcy: K = mu * U_darcy / (dP/L)

    Argumentos:
        df_metrics : DataFrame con resultados de la simulación
        p          : diccionario de parámetros de scaling.py

    Retorna:
        df_metrics con columna K_efectiva añadida
    """
    mu  = p["mu_phy"]
    df_metrics = df_metrics.copy()

    # Convertir U_darcy de lattice a físico
    df_metrics["U_darcy_phy"] = (
        df_metrics["U_darcy"] * p["c_l"] / p["c_t"]
    )

    # Convertir dP_L de lattice a físico
    # Presión LBM: p = rho * cs² = rho / 3
    # dP_L está en [lu/ts²/celda], convertir a [Pa/m]
    cs2    = 1.0 / 3.0
    df_metrics["dP_L_phy"] = (
        df_metrics["dP_L"] * cs2
        * (p["rho_phy"] * p["c_l"] / p["c_t"]**2)
    )

    # Permeabilidad efectiva via Darcy
    df_metrics["K_efectiva"] = (
        mu * df_metrics["U_darcy_phy"] / df_metrics["dP_L_phy"]
    )

    return df_metrics


def compute_errors(df_metrics, p):
    """
    Calcula errores relativos vs soluciones analíticas.

    Retorna df_metrics con columnas de error añadidas.
    """
    df_metrics = df_metrics.copy()

    # Error en permeabilidad vs Kozeny-Carman
    K_KC = p["K_phy"]
    df_metrics["error_K_KC"] = (
        (df_metrics["K_efectiva"] - K_KC) / K_KC
    )

    # Error en caída de presión vs Darcy analítico
    dP_darcy = p["dP_L_darcy"]
    df_metrics["error_dP_darcy"] = (
        (df_metrics["dP_L_phy"] - dP_darcy) / dP_darcy
    )

    # Error en caída de presión vs Ergun
    dP_ergun = p["dP_L_ergun"]
    df_metrics["error_dP_ergun"] = (
        (df_metrics["dP_L_phy"] - dP_ergun) / dP_ergun
    )

    return df_metrics


def compute_seed_statistics(df_metrics):
    """
    Calcula estadísticas entre semillas para un mismo caso.
    Retorna un diccionario con media, std, CV e IC95 de
    las métricas principales.
    """
    cols = ["K_efectiva", "dP_L_phy", "U_darcy_phy",
            "error_K_KC", "error_dP_ergun"]

    stats = {}
    n = len(df_metrics)

    for col in cols:
        if col not in df_metrics.columns:
            continue
        vals        = df_metrics[col].dropna()
        mean        = vals.mean()
        std         = vals.std() if len(vals) > 1 else 0.0
        cv          = std / abs(mean) if mean != 0 else 0.0
        # IC 95% con t-Student
        from scipy import stats as sp_stats
        t_val       = sp_stats.t.ppf(0.975, df=max(len(vals)-1, 1))
        ic95        = t_val * std / np.sqrt(len(vals))

        stats[col] = {
            "mean" : mean,
            "std"  : std,
            "cv"   : cv,
            "ic95" : ic95,
            "n"    : len(vals)
        }

    return stats


# ================================================================
# SECCIÓN 3: SOLUCIONES ANALÍTICAS
# ================================================================

def poiseuille_profile(y_cells, Ny, u_max):
    """
    Perfil de Poiseuille para canal libre (sin obstáculos).
    u(y) = u_max * (1 - ((y - Ly/2) / (Ly/2))²)

    Argumentos:
        y_cells : array de posiciones en celdas
        Ny      : número de celdas en y
        u_max   : velocidad máxima en el centro

    Retorna:
        array de velocidades analíticas
    """
    y_norm = y_cells / Ny          # normalizado 0 a 1
    y_c    = y_norm - 0.5          # centrado en 0
    return u_max * (1.0 - (y_c / 0.5)**2)


def darcy_uniform_profile(y_cells, U_darcy):
    """
    Perfil uniforme de Darcy (velocidad constante en y).
    Solo válido para Re_dp << 1.
    """
    return np.full_like(y_cells, U_darcy, dtype=float)


# ================================================================
# SECCIÓN 4: FIGURAS
# ================================================================

def plot_velocity_profile(df_profile, p, seed=0,
                          show_poiseuille=False,
                          show_darcy=False):
    """
    Grafica el perfil de velocidad Ux promediado en x
    comparado con las soluciones analíticas disponibles.
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    y    = df_profile["y"].values
    ux   = df_profile["ux_mean"].values

    # Perfil LBM
    ax.plot(ux, y, "b-", linewidth=2, label="LBM")

    # Poiseuille (solo canal libre)
    if show_poiseuille:
        u_max     = ux.max()
        Ny        = p["Ny"]
        u_pois    = poiseuille_profile(y, Ny, u_max)
        ax.plot(u_pois, y, "r--", linewidth=1.5,
                label="Poiseuille analítico")

    # Darcy uniforme
    if show_darcy:
        U_darcy = p["u_darcy_lbm"]
        u_darcy = darcy_uniform_profile(y, U_darcy)
        ax.plot(u_darcy, y, "g--", linewidth=1.5,
                label="Darcy uniforme")

    ax.set_xlabel("$u_x$ [lu/ts]")
    ax.set_ylabel("y [celdas]")
    ax.set_title(f"Perfil de velocidad | seed={seed} | φ={p['phi']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dP_vs_Re(results_df, p_list):
    """
    Grafica caída de presión vs Reynolds para múltiples casos.
    Compara LBM vs Darcy vs Ergun.

    Argumentos:
        results_df : DataFrame con columnas Re_dp, dP_L_phy
        p_list     : lista de diccionarios de parámetros
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Puntos LBM
    ax.scatter(results_df["Re_dp"], results_df["dP_L_phy"],
               color="blue", zorder=5, label="LBM")

    # Barras de error entre semillas si hay std
    if "dP_L_std" in results_df.columns:
        ax.errorbar(results_df["Re_dp"], results_df["dP_L_phy"],
                    yerr=results_df["dP_L_std"],
                    fmt="none", color="blue", capsize=4)

    # Curvas analíticas
    Re_range = np.linspace(results_df["Re_dp"].min(),
                           results_df["Re_dp"].max(), 100)

    # Tomar parámetros del primer caso para las curvas analíticas
    p0 = p_list[0]
    dP_darcy_curve = [
        p0["dP_L_darcy"] * (Re / p0["Re_dp"])
        for Re in Re_range
    ]
    dP_ergun_curve = [
        p0["dP_L_ergun"] * (Re / p0["Re_dp"])**2
        for Re in Re_range
    ]

    ax.plot(Re_range, dP_darcy_curve, "r--",
            linewidth=1.5, label="Darcy analítico")
    ax.plot(Re_range, dP_ergun_curve, "g--",
            linewidth=1.5, label="Ergun analítico")

    ax.set_xlabel("$Re_{dp}$")
    ax.set_ylabel("$\\Delta P / L$ [Pa/m]")
    ax.set_title("Caída de presión vs Reynolds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_K_vs_phi(results_df):
    """
    Grafica permeabilidad efectiva vs porosidad.
    Compara K_LBM vs K_Kozeny-Carman.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Puntos LBM
    ax.scatter(results_df["phi"], results_df["K_efectiva"],
               color="blue", zorder=5, label="LBM")

    if "K_std" in results_df.columns:
        ax.errorbar(results_df["phi"], results_df["K_efectiva"],
                    yerr=results_df["K_std"],
                    fmt="none", color="blue", capsize=4)

    # Kozeny-Carman
    if "K_KC" in results_df.columns:
        ax.scatter(results_df["phi"], results_df["K_KC"],
                   color="red", marker="x", zorder=5,
                   label="Kozeny-Carman")

    ax.set_xlabel("Porosidad φ")
    ax.set_ylabel("$K$ [m²]")
    ax.set_title("Permeabilidad efectiva vs Porosidad")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


# ================================================================
# SECCIÓN 5: ACTUALIZAR TABLA MAESTRA
# ================================================================

def update_database(run_dir, p, df_metrics, database_dir):
    """
    Agrega los resultados de esta run a la tabla maestra
    all_results.csv en database_dir.

    Si el archivo no existe lo crea.
    Si ya existe agrega las filas nuevas.
    """
    database_dir = Path(database_dir)
    database_dir.mkdir(parents=True, exist_ok=True)
    fpath = database_dir / "all_results.csv"

    # Agregar columnas de inputs para identificar el caso
    df_out = df_metrics.copy()
    df_out["run_dir"]    = str(run_dir)
    df_out["Re_dp"]      = p["Re_dp"]
    df_out["phi"]        = p["phi"]
    df_out["dp_m"]       = p["dp_m"]
    df_out["Lx_m"]       = p["Lx_m"]
    df_out["Ly_m"]       = p["Ly_m"]
    df_out["Nx"]         = p["Nx"]
    df_out["Ny"]         = p["Ny"]
    df_out["tau"]        = p["tau"]
    df_out["Ma"]         = p["Ma"]
    df_out["K_KC"]       = p["K_phy"]
    df_out["dP_darcy"]   = p["dP_L_darcy"]
    df_out["dP_ergun"]   = p["dP_L_ergun"]

    if fpath.exists():
        df_existing = pd.read_csv(fpath)
        df_final    = pd.concat([df_existing, df_out],
                                ignore_index=True)
    else:
        df_final = df_out

    df_final.to_csv(fpath, index=False)
    print(f"   ✔ Base de datos actualizada: {fpath}")
    print(f"     Total de runs: {len(df_final)}")


# ================================================================
# VERIFICACIÓN RÁPIDA
# ================================================================

if __name__ == "__main__":
    import pandas as pd
    from scaling import compute_params

    p = compute_params(
        Re_dp   = 1.0,
        dp_m    = 0.005,
        phi     = 0.48,
        Lx_m    = 0.10,
        Ly_m    = 0.033,
        rho_phy = 1.204,
        mu_phy  = 1.825e-5,
        tau     = 0.8,
        Nx      = 300
    )

    # Datos simulados de métricas (como si vinieran del CUDA)
    n = 5
    rng = np.random.default_rng(42)

    df_test = pd.DataFrame({
        "seed"      : list(range(n)),
        "t_final"   : [50000] * n,
        "converged" : [True] * n,
        "U_darcy"   : p["u_darcy_lbm"] * (1 + 0.02 * rng.standard_normal(n)),
        "dP_L"      : (p["u_darcy_lbm"] * p["nu_lbm"] / p["K_lbm"])
                      * (1 + 0.03 * rng.standard_normal(n)),
    })

    # Calcular métricas
    df_test = compute_K_effective(df_test, p)
    df_test = compute_errors(df_test, p)

    # Estadísticas entre semillas
    stats = compute_seed_statistics(df_test)

    print("\n── Resultados por semilla ──────────────────────────")
    print(df_test[["seed", "K_efectiva",
                   "dP_L_phy", "error_K_KC"]].to_string())

    print("\n── Estadísticas entre semillas ─────────────────────")
    for col, s in stats.items():
        print(f"  {col}:")
        print(f"    mean = {s['mean']:.4e}")
        print(f"    std  = {s['std']:.4e}")
        print(f"    CV   = {s['cv']*100:.2f}%")
        print(f"    IC95 = ± {s['ic95']:.4e}")

    print("\n✔ postprocess.py verificado correctamente.")