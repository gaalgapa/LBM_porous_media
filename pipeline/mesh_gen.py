# ================================================================
# mesh_gen.py
# Genera las máscaras binarias de obstáculos para la simulación.
# Cada máscara es un array 2D donde 1 = sólido, 0 = fluido.
# ================================================================

import numpy as np
from pathlib import Path


# ================================================================
# SECCIÓN 1: GENERADORES DE GEOMETRÍA
# Cada función retorna una lista de círculos (cx, cy, r) en [m]
# ================================================================

def gen_single_sphere(Lx, Ly, dp):
    """Una sola esfera centrada en el canal."""
    r = dp / 2
    return [(Lx / 2, Ly / 2, r)]


def gen_three_vertical(Lx, Ly, dp):
    """Tres esferas alineadas verticalmente en el centro."""
    r       = dp / 2
    cx      = Lx / 2
    cy      = Ly / 2
    spacing = dp * 1.5
    circles = [(cx, cy, r),
               (cx, cy + spacing, r),
               (cx, cy - spacing, r)]
    # Filtrar las que salgan del dominio
    circles = [c for c in circles if r <= c[1] <= Ly - r]
    return circles


def gen_nine_block(Lx, Ly, dp):
    """Nueve esferas en bloque 3x3 centrado en el canal."""
    r       = dp / 2
    cx      = Lx / 2
    cy      = Ly / 2
    spacing = dp * 1.5
    offsets = [-spacing, 0, spacing]
    circles = [(cx + dx, cy + dy, r)
               for dx in offsets
               for dy in offsets]
    # Filtrar las que salgan del dominio
    circles = [c for c in circles
               if (r <= c[0] <= Lx - r) and (r <= c[1] <= Ly - r)]
    return circles


def gen_packed_bed(Lx, Ly, dp, phi_target, rng):
    """
    Lecho empacado aleatorio.
    Coloca partículas aleatoriamente hasta alcanzar la porosidad objetivo.
    Este es el único generador válido para comparar con Kozeny-Carman.
    """
    circles    = []
    area_solid = 0.0
    r          = dp / 2
    x_min, x_max = r, Lx - r
    y_min, y_max = r, Ly - r
    max_iter   = 10000

    if x_max <= x_min or y_max <= y_min:
        print("❌ Partículas demasiado grandes para el dominio.")
        return circles

    attempts = 0
    while attempts < max_iter:
        # Verificar si ya alcanzamos la porosidad objetivo
        phi_actual = 1 - area_solid / (Lx * Ly)
        if phi_actual <= phi_target:
            break

        # Intentar colocar una nueva partícula
        x = rng.random() * (x_max - x_min) + x_min
        y = rng.random() * (y_max - y_min) + y_min

        # Verificar que no se solape con ninguna existente
        no_overlap = all(
            np.hypot(x - cx, y - cy) >= dp
            for (cx, cy, cr) in circles
        )

        if no_overlap:
            circles.append((x, y, r))
            area_solid += np.pi * r**2

        attempts += 1

    phi_final = 1 - area_solid / (Lx * Ly)
    if phi_final > phi_target:
        print(f"   ⚠️  Porosidad objetivo no alcanzada.")
        print(f"   φ objetivo: {phi_target:.4f} | φ final: {phi_final:.4f}")
    else:
        print(f"   ✔ Partículas: {len(circles)} | φ = {phi_final:.4f}")

    return circles


# ================================================================
# SECCIÓN 2: RASTERIZACIÓN
# Convierte la lista de círculos a un array 2D de 0s y 1s
# ================================================================

def rasterize(circles, Nx, Ny, c_l):
    """
    Convierte lista de círculos (cx, cy, r) en metros
    a una máscara binaria de Nx x Ny celdas.
    1 = sólido (obstáculo), 0 = fluido
    """
    mask = np.zeros((Nx, Ny), dtype=np.uint8)

    # Coordenadas físicas del centro de cada celda
    X = np.arange(Nx) * c_l   # shape (Nx,)
    Y = np.arange(Ny) * c_l   # shape (Ny,)
    X, Y = np.meshgrid(X, Y, indexing='ij')  # shape (Nx, Ny)

    for (cx, cy, r) in circles:
        mask |= ((X - cx)**2 + (Y - cy)**2 <= r**2).astype(np.uint8)

    return mask


# ================================================================
# SECCIÓN 3: FUNCIÓN PRINCIPAL
# Genera N máscaras para un set de parámetros dado
# ================================================================

def generate_masks(p, run_dir, geometry=3, n_seeds=5):
    """
    Genera n_seeds máscaras aleatorias para el caso definido en p.

    Argumentos:
        p         : diccionario de parámetros de scaling.py
        run_dir   : Path donde guardar las máscaras
        geometry  : 0=una esfera | 1=tres verticales |
                    2=nueve bloque | 3=lecho empacado
        n_seeds   : número de mallas aleatorias a generar

    Retorna:
        lista de rutas a los archivos .bin generados
    """
    run_dir = Path(run_dir)
    mask_dir = run_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    Nx  = p["Nx"]
    Ny  = p["Ny"]
    c_l = p["c_l"]
    Lx  = p["Lx_m"]
    Ly  = p["Ly_m"]
    dp  = p["dp_m"]
    phi = p["phi"]

    mask_files = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        # Seleccionar generador
        if geometry == 0:
            circles = gen_single_sphere(Lx, Ly, dp)
            geo_name = "single"
        elif geometry == 1:
            circles = gen_three_vertical(Lx, Ly, dp)
            geo_name = "three_vertical"
        elif geometry == 2:
            circles = gen_nine_block(Lx, Ly, dp)
            geo_name = "nine_block"
        else:
            circles = gen_packed_bed(Lx, Ly, dp, phi, rng)
            geo_name = "packed_bed"

        # Rasterizar
        mask = rasterize(circles, Nx, Ny, c_l)

        # Porosidad real de la máscara
        phi_actual = 1.0 - mask.sum() / (Nx * Ny)

        # Nombre del archivo
        fname = (f"mask_{geo_name}"
                 f"_phi{phi_actual:.4f}"
                 f"_seed{seed}"
                 f"_{Nx}x{Ny}.bin")
        fpath = mask_dir / fname

        # Guardar
        mask.astype(np.uint8).tofile(fpath)

        print(f"   💾 seed={seed} | φ_real={phi_actual:.4f}"
              f" | {Nx}x{Ny} | {fname}")

        mask_files.append(fpath)

    return mask_files


# ================================================================
# SECCIÓN 4: UTILIDADES
# ================================================================

def load_mask(fpath):
    """Carga una máscara binaria desde disco."""
    fpath = Path(fpath)
    # El nombre contiene NxNy, lo extraemos para reconstruir el shape
    # Formato: ..._NxNy.bin  ej: ..._300x100.bin
    stem  = fpath.stem          # sin extensión
    dims  = stem.split("_")[-1] # último token: "300x100"
    Nx, Ny = map(int, dims.split("x"))
    data  = np.fromfile(fpath, dtype=np.uint8)
    return data.reshape(Nx, Ny)


def porosity_from_mask(mask):
    """Calcula la porosidad real de una máscara."""
    return 1.0 - mask.sum() / mask.size

def plot_mask(mask, title=""):
    """Visualiza una máscara binaria."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mask.T, origin="lower", cmap="gray_r",
              aspect="equal",
              extent=[0, mask.shape[0], 0, mask.shape[1]])
    ax.set_xlabel("x [celdas]")
    ax.set_ylabel("y [celdas]")
    ax.set_title(title or "Máscara de obstáculos")
    plt.tight_layout()
    plt.show()


# ================================================================
# VERIFICACIÓN RÁPIDA
# ================================================================

if __name__ == "__main__":
    from scaling import compute_params
    from paths import get_runs_dir, new_run_name

    # Parámetros de prueba
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

    # Carpeta de prueba
    run_dir = get_runs_dir() / new_run_name("prueba_mesh")

    print(f"\nGenerando máscaras en: {run_dir}\n")

    # Generar 3 semillas de lecho empacado
    archivos = generate_masks(p, run_dir, geometry=3, n_seeds=3)

    print(f"\n✔ {len(archivos)} máscaras generadas.")

    # Verificar que se pueden cargar
    mask = load_mask(archivos[0])
    phi  = porosity_from_mask(mask)
    print(f"  Verificación carga: shape={mask.shape} | φ={phi:.4f}")
    
    plot_mask(mask, title=f"Lecho empacado | φ={phi:.4f} | seed=0")