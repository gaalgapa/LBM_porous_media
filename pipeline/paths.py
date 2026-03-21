# ================================================================
# Define las rutas del proyecto en un solo lugar
# ================================================================

from pathlib import Path

# ── Raíz del proyecto (donde está este archivo) ─────────────────
# Path(__file__) es la ruta de paths.py
# .parent sube un nivel → llega a pipeline/
# .parent otra vez → llega a lbm-porous-media/
PROJECT_ROOT = Path(__file__).parent.parent

# ── Código fuente ────────────────────────────────────────────────
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
CUDA_DIR     = PROJECT_ROOT / "cuda"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# ── Ejecutable compilado (se genera en Colab) ───────────────────
EXECUTABLE   = CUDA_DIR / "lbm_sim"

# ── Datos en Google Drive ────────────────────────────────────────
# En tu PC: Drive accesible desde el explorador de archivos
# En Colab: /content/drive/MyDrive/
# La función get_data_root() detecta automáticamente dónde estás

def get_data_root() -> Path:
    """
    Devuelve la raíz de datos según el entorno donde se ejecuta.
    - En Colab: /content/drive/MyDrive/LBM_Experiments
    - En PC local: ~/GoogleDrive/LBM_Experiments (si existe el enlace)
    - Fallback: ~/LBM_Experiments (crea la carpeta localmente)
    """
    # Colab
    colab_path = Path("/content/drive/MyDrive/LBM_Experiments")
    if colab_path.exists():
        return colab_path

    # PC local con enlace simbólico a Drive
    local_drive = Path.home() / "GoogleDrive" / "LBM_Experiments"
    if local_drive.exists():
        return local_drive

    # Fallback: carpeta local sin Drive
    fallback = PROJECT_ROOT / "LBM_Experiments"
    fallback.mkdir(parents=True, exist_ok=True)
    print(f"⚠️  Drive no encontrado. Usando carpeta local: {fallback}")
    return fallback


# ── Subcarpetas de datos ─────────────────────────────────────────
def get_runs_dir() -> Path:
    path = get_data_root() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_database_dir() -> Path:
    path = get_data_root() / "database"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Utilidad: nombre de run con fecha y hora ─────────────────────
from datetime import datetime

def new_run_name(label: str = "") -> str:
    """
    Genera un nombre único para una run basado en fecha y hora.
    Ejemplo: '2026-03-20_14-32_canal-libre'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if label:
        # Reemplaza espacios por guiones para nombres de carpeta limpios
        label_clean = label.strip().replace(" ", "-").lower()
        return f"{timestamp}_{label_clean}"
    return timestamp


# ── Verificación rápida al importar ─────────────────────────────
if __name__ == "__main__":
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"PIPELINE_DIR : {PIPELINE_DIR}")
    print(f"CUDA_DIR     : {CUDA_DIR}")
    print(f"EXECUTABLE   : {EXECUTABLE}")
    print(f"DATA_ROOT    : {get_data_root()}")
    print(f"RUNS_DIR     : {get_runs_dir()}")
    print(f"DATABASE_DIR : {get_database_dir()}")
    print(f"Run name     : {new_run_name('prueba canal')}")