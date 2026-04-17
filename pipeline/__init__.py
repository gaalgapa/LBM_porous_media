# ================================================================
# __init__.py
# Hace que pipeline/ sea un módulo Python importable.
# Expone las funciones principales de cada módulo.
# ================================================================

from .scaling   import compute_params, is_stable, print_summary
from .mesh_gen  import generate_masks, load_mask, plot_mask
from .postprocess import (compute_K_effective, compute_errors,
                           compute_seed_statistics, update_database)
from .paths     import (get_data_root, get_runs_dir,
                         get_database_dir, new_run_name)