from .magnetic_models import (
    TopologicalInsulatorModel,
    TopologicalInsulatorParams,
    build_lattice_pairs,
    build_lattice_pairs_periodic,
    calculate_chemical_potential,

)
from .utils import (
    plot_interactive_mu_c,
    plot_interactive_mu_dist,
)

__version__ = "0.1.2"