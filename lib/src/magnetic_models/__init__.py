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

from importlib.metadata import version as _version

__version__ = _version("magnetic_models")