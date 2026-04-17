# ================================================================
# master.py
# Orquestador principal del pipeline LBM.
#
# Modos de uso:
#   python master.py --single --Re 1 --phi 0.48 --dp 0.005
#   python master.py --batch cases.csv
#   python master.py --sweep --Re_range 1 150 10 \
#                             --phi_range 0.35 0.80 0.05
#   python master.py --rerun run_00042
# ================================================================

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from paths      import get_runs_dir, get_database_dir, new_run_name
from scaling    import compute_params, is_stable, print_summary
from mesh_gen   import generate_masks
from postprocess import (compute_K_effective, compute_errors,
                          compute_seed_statistics, update_database)


# ================================================================
# SECCIÓN 1: PARSEO DE ARGUMENTOS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline LBM D2Q9 para flujo en medios porosos"
    )

    # Modo de ejecución
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true",
                      help="Correr un caso específico")
    mode.add_argument("--batch",  type=str, metavar="CSV",
                      help="Correr lista de casos desde CSV")
    mode.add_argument("--sweep",  action="store_true",
                      help="Barrido paramétrico completo")
    mode.add_argument("--rerun",  type=str, metavar="RUN_DIR",
                      help="Repetir un caso anterior")

    # Parámetros físicos para --single
    parser.add_argument("--Re",  type=float, help="Reynolds dp")
    parser.add_argument("--phi", type=float, help="Porosidad")
    parser.add_argument("--dp",  type=float, help="Diámetro partícula [m]")
    parser.add_argument("--Lx",  type=float, default=0.10,
                        help="Longitud canal [m]")
    parser.add_argument("--Ly",  type=float, default=0.033,
                        help="Altura canal [m]")
    parser.add_argument("--rho", type=float, default=1.204,
                        help="Densidad fluido [kg/m3]")
    parser.add_argument("--mu",  type=float, default=1.825e-5,
                        help="Viscosidad dinámica [Pa·s]")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Parámetro de relajación LBM")
    parser.add_argument("--Nx",  type=int,   default=300,
                        help="Celdas en x")

    # Opciones para --single y --sweep
    parser.add_argument("--seeds",    type=int, default=5,
                        help="Número de semillas por caso")
    parser.add_argument("--geometry", type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help="0=esfera 1=tres 2=bloque 3=empacado")
    parser.add_argument("--label",    type=str, default="",
                        help="Etiqueta descriptiva del caso")

    # Rangos para --sweep
    parser.add_argument("--Re_range",  nargs=3, type=float,
                        metavar=("MIN","MAX","STEP"),
                        help="Rango de Reynolds")
    parser.add_argument("--phi_range", nargs=3, type=float,
                        metavar=("MIN","MAX","STEP"),
                        help="Rango de porosidad")
    parser.add_argument("--dp_list",   nargs="+", type=float,
                        help="Lista de diámetros [m]")

    return parser.parse_args()


# ================================================================
# SECCIÓN 2: CONSTRUCCIÓN DE LA LISTA DE CASOS
# ================================================================

def build_cases_single(args):
    """Un solo caso definido en CLI."""
    if not all([args.Re, args.phi, args.dp]):
        print("❌ --single requiere --Re, --phi y --dp")
        sys.exit(1)
    return [{
        "Re_dp"   : args.Re,
        "phi"     : args.phi,
        "dp_m"    : args.dp,
        "Lx_m"    : args.Lx,
        "Ly_m"    : args.Ly,
        "rho_phy" : args.rho,
        "mu_phy"  : args.mu,
        "tau"     : args.tau,
        "Nx"      : args.Nx,
        "n_seeds" : args.seeds,
        "geometry": args.geometry,
        "label"   : args.label or f"Re{args.Re}_phi{args.phi}",
    }]


def build_cases_batch(csv_path):
    """Lista de casos desde un CSV."""
    df = pd.read_csv(csv_path)
    cases = []
    for _, row in df.iterrows():
        cases.append({
            "Re_dp"   : float(row["Re"]),
            "phi"     : float(row["phi"]),
            "dp_m"    : float(row["dp_m"]),
            "Lx_m"    : float(row.get("Lx_m",  0.10)),
            "Ly_m"    : float(row.get("Ly_m",  0.033)),
            "rho_phy" : float(row.get("rho_phy", 1.204)),
            "mu_phy"  : float(row.get("mu_phy",  1.825e-5)),
            "tau"     : float(row.get("tau",  0.8)),
            "Nx"      : int(row.get("Nx", 300)),
            "n_seeds" : int(row.get("n_seeds", 5)),
            "geometry": int(row.get("geometry", 3)),
            "label"   : str(row.get("notes", "")),
        })
    return cases


def build_cases_sweep(args):
    """Producto cartesiano de rangos."""
    if not args.Re_range or not args.phi_range or not args.dp_list:
        print("❌ --sweep requiere --Re_range, --phi_range y --dp_list")
        sys.exit(1)

    Re_list  = np.arange(args.Re_range[0],
                          args.Re_range[1] + args.Re_range[2]/2,
                          args.Re_range[2])
    phi_list = np.arange(args.phi_range[0],
                          args.phi_range[1] + args.phi_range[2]/2,
                          args.phi_range[2])
    dp_list  = args.dp_list

    cases = []
    for Re in Re_list:
        for phi in phi_list:
            for dp in dp_list:
                cases.append({
                    "Re_dp"   : float(Re),
                    "phi"     : float(phi),
                    "dp_m"    : float(dp),
                    "Lx_m"    : args.Lx,
                    "Ly_m"    : args.Ly,
                    "rho_phy" : args.rho,
                    "mu_phy"  : args.mu,
                    "tau"     : args.tau,
                    "Nx"      : args.Nx,
                    "n_seeds" : args.seeds,
                    "geometry": args.geometry,
                    "label"   : f"Re{Re:.1f}_phi{phi:.2f}_dp{dp:.4f}",
                })
    return cases


def build_cases_rerun(run_dir):
    """Lee config.json de un caso anterior."""
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        print(f"❌ No se encontró config.json en {run_dir}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    return [cfg]


# ================================================================
# SECCIÓN 3: GUARDAR CONFIG
# ================================================================

def save_config(p, run_dir, geometry, n_seeds):
    """Guarda config.json con todos los parámetros del caso."""
    config = dict(p)   # copia del diccionario de scaling
    config["geometry"] = geometry
    config["n_seeds"]  = n_seeds
    config["save_every"]  = max(1000, p["t_max"] // 10)
    config["ckpt_every"]  = max(1000, p["t_max"] // 20)
    config["max_diff"]    = 1.0e-8

    config_path = Path(run_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


# ================================================================
# SECCIÓN 4: EJECUTAR UN CASO
# ================================================================

def run_case(case_params):
    """
    Ejecuta el pipeline completo para un caso:
    1. Scaling y validación
    2. Generar mallas
    3. Correr simulación CUDA para cada seed
    4. Postprocesar y actualizar base de datos
    """
    from paths import PROJECT_ROOT

    # ── Calcular parámetros ──────────────────────────────────────
    p = compute_params(
        Re_dp   = case_params["Re_dp"],
        dp_m    = case_params["dp_m"],
        phi     = case_params["phi"],
        Lx_m    = case_params["Lx_m"],
        Ly_m    = case_params["Ly_m"],
        rho_phy = case_params["rho_phy"],
        mu_phy  = case_params["mu_phy"],
        tau     = case_params["tau"],
        Nx      = case_params["Nx"],
    )

    # ── Validar estabilidad ──────────────────────────────────────
    stable, errors, warnings = is_stable(p)

    for w in warnings:
        print(f"  {w}")

    if not stable:
        for e in errors:
            print(f"  {e}")
        print(f"  ⏭️  Caso omitido: Re={p['Re_dp']} "
              f"phi={p['phi']} dp={p['dp_m']}")
        return None

    print_summary(p, label=case_params.get("label",""))

    # ── Crear carpeta de la run ──────────────────────────────────
    label    = case_params.get("label", "")
    run_name = new_run_name(label)
    run_dir  = get_runs_dir() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    geometry = case_params.get("geometry", 3)
    n_seeds  = case_params.get("n_seeds",  5)

    # ── Guardar config ───────────────────────────────────────────
    config_path = save_config(p, run_dir, geometry, n_seeds)
    print(f"  💾 Config guardado: {config_path}")

    # ── Generar mallas ───────────────────────────────────────────
    print(f"\n  Generando {n_seeds} mallas...")
    mask_files = generate_masks(p, run_dir,
                                 geometry=geometry,
                                 n_seeds=n_seeds)

    # ── Ejecutable CUDA ──────────────────────────────────────────
    executable = PROJECT_ROOT / "cuda" / "lbm_sim"
    if not executable.exists():
        print(f"❌ Ejecutable no encontrado: {executable}")
        print("   Compila primero con: nvcc -o cuda/lbm_sim ...")
        return None

    # ── Correr simulación para cada seed ────────────────────────
    print(f"\n  Corriendo simulación ({n_seeds} seeds)...")
    for seed, mask_path in enumerate(mask_files):
        print(f"\n  ── Seed {seed} ──────────────────────────────")
        result = subprocess.run(
            [str(executable),
             str(config_path),
             str(mask_path),
             str(seed)],
            capture_output=False,   # mostrar output en tiempo real
            timeout=86400           # 24 horas máximo
        )
        if result.returncode != 0:
            print(f"  ❌ Seed {seed} falló con código "
                  f"{result.returncode}")

    # ── Postprocesar ─────────────────────────────────────────────
    print(f"\n  Postprocesando resultados...")
    metrics_path = run_dir / "results" / "metrics.csv"

    if not metrics_path.exists():
        print("  ⚠️  metrics.csv no encontrado, "
              "simulación no completada.")
        return None

    import pandas as pd
    df = pd.read_csv(metrics_path)
    df = compute_K_effective(df, p)
    df = compute_errors(df, p)

    stats = compute_seed_statistics(df)

    print("\n  ── Estadísticas entre seeds ─────────────────")
    for col, s in stats.items():
        print(f"    {col}: {s['mean']:.4e} ± {s['ic95']:.4e} "
              f"(CV={s['cv']*100:.1f}%)")

    # ── Actualizar base de datos ─────────────────────────────────
    update_database(run_dir, p, df, get_database_dir())

    return run_dir


# ================================================================
# SECCIÓN 5: MAIN
# ================================================================

def main():
    args = parse_args()

    # Construir lista de casos según el modo
    if args.single:
        cases = build_cases_single(args)

    elif args.batch:
        cases = build_cases_batch(args.batch)

    elif args.sweep:
        cases = build_cases_sweep(args)

    elif args.rerun:
        cases = build_cases_rerun(args.rerun)

    print(f"\n{'='*54}")
    print(f"  Total de casos a procesar: {len(cases)}")
    print(f"{'='*54}\n")

    # Filtrar casos inválidos antes de correr
    valid_cases = []
    print("Validando casos...")
    for c in cases:
        p = compute_params(
            Re_dp   = c["Re_dp"],
            dp_m    = c["dp_m"],
            phi     = c["phi"],
            Lx_m    = c["Lx_m"],
            Ly_m    = c["Ly_m"],
            rho_phy = c["rho_phy"],
            mu_phy  = c["mu_phy"],
            tau     = c["tau"],
            Nx      = c["Nx"],
        )
        stable, errors, _ = is_stable(p)
        if stable:
            valid_cases.append(c)
        else:
            print(f"  ⏭️  Omitido Re={c['Re_dp']} "
                  f"phi={c['phi']}: {errors[0]}")

    print(f"\n  Casos válidos: {len(valid_cases)} / {len(cases)}")

    if not valid_cases:
        print("❌ No hay casos válidos para correr.")
        sys.exit(1)

    # Correr todos los casos válidos
    results = []
    for i, case in enumerate(valid_cases):
        print(f"\n{'='*54}")
        print(f"  Caso {i+1}/{len(valid_cases)}: "
              f"Re={case['Re_dp']} "
              f"phi={case['phi']} "
              f"dp={case['dp_m']}")
        print(f"{'='*54}")

        run_dir = run_case(case)
        if run_dir:
            results.append(str(run_dir))

    print(f"\n{'='*54}")
    print(f"  Completado: {len(results)}/{len(valid_cases)} casos")
    print(f"  Resultados en: {get_database_dir()}/all_results.csv")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()