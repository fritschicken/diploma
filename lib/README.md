# magnetic_models

A tight-binding implementation of a 2D topological insulator (TI) coupled to a magnetic layer, built on [sisl](https://github.com/zerothi/sisl).

Each unit cell has 6 orbitals:

- 4 orbitals for the TI layer: the BHZ model with spin-orbit-coupled s- and p-like states
- 2 orbitals for the magnetic layer: spin up and spin down, in an exchange field **b**

The package builds the geometry, Hamiltonian and density matrix. These can then be passed to tools such as [grogupy](https://pypi.org/project/grogupy/) to calculate exchange interactions.

## Installation

```bash
pip install "git+https://github.com/fritschicken/diploma.git#subdirectory=lib"
```

Requires Python ≥ 3.10.

## Quick start

```python
from magnetic_models import TopologicalInsulatorModel, TopologicalInsulatorParams

params = TopologicalInsulatorParams(u=-1.2, b=(0.0, 0.0, 1.0), c_inter=0.5)

model = TopologicalInsulatorModel()
H, DM = model.automate(params, Nx=10, Ny=10, a=1.0, z=10.0, kT=0.01)

print(model.magnetic_moment())
```

`automate` runs the full pipeline. You can also run each step yourself:

```python
model = TopologicalInsulatorModel()
model.build_geometry(Nx=10, Ny=10, a=1.0, z=10.0, abc=[5, 2, 5])  # sisl boundary codes
model.build_unit_cell(params)          # 6x6 onsite (U) and hopping (Tx, Ty) matrices
H = model.build_hamiltonian()          # sisl.Hamiltonian
DM = model.build_density_matrix(kT=0.01, fermi_level=0.0)
```

## Model parameters

`TopologicalInsulatorParams` fields (energies in eV):

| Field | Default | Meaning |
| --- | --- | --- |
| `u` | -1.2 | TI mass term (onsite potential) |
| `t_ti` | 0.5 | TI normal hopping |
| `v_ti` | 0.5 | TI spin-orbit hopping (Dirac velocity) |
| `c_intra` | 0.0 | Inter-orbital coupling inside the TI |
| `typ` | `"asym"` | Coupling symmetry, `"sym"` or `"asym"` |
| `b` | (1, 0, 0) | Exchange field (bx, by, bz) on the magnetic layer |
| `t_x`, `t_y` | -1.0 | Hopping in the magnetic layer |
| `c_inter` | 0.0 | Coupling between the TI and the magnetic layer |
| `mu` | 0.0 | Chemical potential shift of the magnetic layer |

## API

| Name | Description |
| --- | --- |
| `TopologicalInsulatorModel` | Builds the geometry, Hamiltonian and density matrix |
| `TopologicalInsulatorParams` | Dataclass holding the physical parameters |
| `calculate_chemical_potential(H, filling_per_cell, kT)` | Chemical potential for a given filling per unit cell |
| `build_lattice_pairs(Nx, Ny, radius, atoms_per_cell)` | Neighbour pairs within `radius` on an open lattice |
| `build_lattice_pairs_periodic(Nx, Ny, radius, atom_indices)` | Neighbour pairs with periodic boundaries in x and y |
| `plot_interactive_mu_c`, `plot_interactive_mu_dist` | Interactive Plotly views of exchange parameters from saved `.npz` results |

## License

MIT. See [LICENSE](LICENSE).
