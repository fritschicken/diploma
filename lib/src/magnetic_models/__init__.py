from .magnetic_models import (
    TopologicalInsulatorModel,
    TopologicalInsulatorParams,
    build_lattice_pairs,
    build_lattice_pairs_periodic,
    calculate_chemical_potential,
)
from .utils import (
    Sweep,
    load_sweep,
    interactive_dual_heatmap,
    plot_interactive_mu_c,
    plot_interactive_mu_dist,
    plot_heatmap_mu_cinter,
)

from importlib.metadata import version as _version

__version__ = _version("magnetic_models")