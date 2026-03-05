from __future__ import annotations
import numpy as np
import sisl as si
from dataclasses import dataclass
from typing import Optional, List, Tuple

__all__ = ['MagneticModel', 'MagneticParams', 'TopologicalInsulatorModel', 'TopologicalInsulatorParams','build_lattice_pairs','build_lattice_pairs_periodic','calculate_chemical_potential','generate_custom_matrices']

# =============================================================================
# topological_insulator_model.py
# =============================================================================
# A minimal tight-binding implementation of a 2D Topological Insulator (TI)
# coupled to a magnetic layer using the sisl package.
#
# Author : Laszlo Erik Fritsch
# License: MIT
# =============================================================================

# -----------------------------------------------------------------------------
# Dataclass: TopologicalInsulatorParams
# -----------------------------------------------------------------------------
@dataclass
class TopologicalInsulatorParams:
    """
    Container for all physical model parameters used to construct the unit cell.

    Parameters
    ----------
    u : float
        Onsite potential strength (mass term) of the topological-insulator (TI) block.
        Default is -1.2 eV.
    c_intra : float
        Inter-orbital coupling strength within the TI block.
        Default is 0 eV.
    typ : str
        Coupling symmetry type between orbitals. Options are 'sym' (symmetric) 
        or 'asym' (asymmetric). Default is 'asym'.
    b : tuple of float
        Magnetic exchange field vector (bx, by, bz) in eV applied to the magnetic layer.
        Default is (1.0, 0.0, 0.0).
    t_x : float
        Hopping amplitude along the x-direction within the magnetic layer.
        Default is -1.0 eV.
    t_y : float
        Hopping amplitude along the y-direction within the magnetic layer.
        Default is -1.0 eV.
    c_inter : float
        Coupling strength between the magnetic and TI layers.
        Default is 0.0 eV.
    mu : float
        Chemical potential shift for the magnetic block.
        Default is 0.0 eV.
    """
    u: float = -1.2
    c_intra: float = 0
    typ: str = "asym"
    b: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    t_x: float = -1.0
    t_y: float = -1.0
    c_inter: float = 0.0
    mu: float = 0.0


# -----------------------------------------------------------------------------
# Class: TopologicalInsulatorModel
# -----------------------------------------------------------------------------
class TopologicalInsulatorModel:
    """
    A 2D tight-binding model for a Topological Insulator (TI) coupled to a magnetic layer.

    This class manages the geometry, Hamiltonian construction, and density matrix 
    calculations using the `sisl` library. The basis consists of 6 orbitals per unit cell:
    - 4 orbitals for the TI layer (spin-orbit coupled s- and p-like states).
    - 2 orbitals for the Magnetic layer (spin-up and spin-down).

    Attributes
    ----------
    Nx, Ny : int
        Number of unit cell repetitions in the x and y directions.
    a : float
        In-plane lattice constant.
    z : float
        Interlayer spacing between the TI and Magnetic layers.
    geom : sisl.Geometry
        The sisl Geometry object defining atoms and lattice.
    H : sisl.Hamiltonian
        The constructed tight-binding Hamiltonian.
    DM : sisl.DensityMatrix
        The calculated density matrix.
    U, Tx, Ty : np.ndarray
        Cached 6x6 matrices for Onsite, Hopping-X, and Hopping-Y terms.
    """

    # Constants for block slicing
    TI_SIZE = 4
    MAG_SIZE = 2
    CELL_SIZE = TI_SIZE + MAG_SIZE

    # Pre-computed Pauli matrices and Identity
    _SX = np.array([[0, 1], [1, 0]], dtype=complex)
    _SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _SZ = np.array([[1, 0], [0, -1]], dtype=complex)
    _S0 = np.eye(2, dtype=complex)

    def __init__(self) -> None:
        self.Nx: Optional[int] = None
        self.Ny: Optional[int] = None
        self.a: Optional[float] = None
        self.z: Optional[float] = None
        
        self.geom: Optional[si.Geometry] = None
        self.H: Optional[si.Hamiltonian] = None
        self.DM: Optional[si.DensityMatrix] = None
        
        # Matrix caches
        self.U: Optional[np.ndarray] = None
        self.Tx: Optional[np.ndarray] = None
        self.Ty: Optional[np.ndarray] = None

        # Atom definitions
        self.Top_ins: Optional[si.Atom] = None
        self.Magnetic: Optional[si.Atom] = None

    # -------------------------------------------------------------------------
    # Geometry Construction
    # -------------------------------------------------------------------------
    def build_geometry(
        self,
        Nx: int,
        Ny: int,
        a: float,
        z: float,
        abc: Optional[List[int]] = None,
        geom: Optional[si.Geometry] = None,
    ) -> si.Geometry:
        """
        Builds or assigns the geometry for the bilayer system.

        Parameters
        ----------
        Nx, Ny : int
            Number of unit cells to tile along the x and y directions.
        a : float
            Lattice constant for the x and y directions (Angstroms).
        z : float
            Interlayer distance between TI and Magnetic layers (Angstroms).
        abc : list of int, optional
            Boundary condition codes for the lattice vectors [A, B, C].
            Common sisl codes: 1=Open, 5=Periodic. Default is [5, 2, 5].
        geom : sisl.Geometry, optional
            An existing geometry object. If provided, it overrides the internal construction.

        Returns
        -------
        sisl.Geometry
            The constructed geometry object.

        Raises
        ------
        ValueError
            If z <= a (interlayer spacing usually must be larger than in-plane spacing for this model).
        """
        self.Nx, self.Ny, self.a, self.z = Nx, Ny, a, z

        # Validate physical spacing
        if z <= a:
            raise ValueError(
                f"Invalid geometry: interlayer spacing z ({z}) must be greater than "
                f"the in-plane lattice constant a ({a})."
            )

        # Use external geometry if provided
        if geom is not None:
            self.geom = geom
            return geom

        # Default boundary conditions
        if abc is None:
            abc = [5, 2, 5]

        # Calculate number of supercells (nsc) for sisl based on BCs
        # If code is 2 (Dirichlet/Custom), use 3 supercells, otherwise 1
        nsc = [3 if code == 2 else 1 for code in abc]

        # Define Atomic Species & Orbitals
        # TI Atom: Carbon-like (Z=6), with spherical + pz orbitals
        r = np.linspace(0, 3, 200)
        f = np.exp(-2 * r**2)
        sorb = si.SphericalOrbital(1, (r, f), R={"contains": 0.99})
        aorb = si.AtomicOrbital("pz", spherical=sorb)
        
        self.Top_ins = si.Atom(6, [sorb, aorb])  
        self.Magnetic = si.Atom(1)  # Hydrogen-like (Z=1)

        # Define Lattice
        lattice = si.Lattice([a, a, z], nsc=nsc)
        lattice.set_boundary_condition(a=abc[0], b=abc[1], c=abc[2])

        # Define Atoms and Coordinates (Basis)
        # Atom 1 (TI) at [a/2, a/2, 0]
        # Atom 2 (Mag) at [a/2, a/2, z]
        base_geom = si.Geometry(
            [[a / 2.0, a / 2.0, 0.0], [a / 2.0, a / 2.0, z]],
            atoms=[self.Top_ins, self.Magnetic],
            lattice=lattice,
        )

        # Create Supercell via Tiling
        self.geom = base_geom.tile(Nx, 0).tile(Ny, 1)
        return self.geom

    # -------------------------------------------------------------------------
    # Unit Cell Matrix Construction
    # -------------------------------------------------------------------------
    def build_unit_cell(
        self, params: Optional[TopologicalInsulatorParams] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the 6x6 tight-binding matrices for the unit cell.

        This method calculates the Onsite (U), Hopping-X (Tx), and Hopping-Y (Ty) 
        matrices based on the provided physical parameters.

        Parameters
        ----------
        params : TopologicalInsulatorParams, optional
            Dataclass instance containing model parameters. If None, a new instance 
            is created using defaults or `kwargs`.
        **kwargs : dict
            Keyword arguments to override specific parameters in `params`.

        Returns
        -------
        U, Tx, Ty : tuple of np.ndarray
            The onsite interaction matrix and nearest-neighbor hopping matrices.
        """

        # Initialize parameters
        if params is None:
            params = TopologicalInsulatorParams(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(params, k):
                    setattr(params, k, v)

        # Unpack parameters for readability
        u, c_intra, typ = params.u, params.c_intra, params.typ
        b, t_x, t_y = params.b, params.t_x, params.t_y
        c_inter, mu = params.c_inter, params.mu

        sx, sy, sz, s0 = self._SX, self._SY, self._SZ, self._S0

        # --- TI Block (4x4) ---
        # Inter-orbital coupling term
        if typ == "sym":
            coup = np.kron(sx, c_intra * sx)
        elif typ == "asym":
            coup = np.kron(sy, c_intra * sx)
        else:
            raise ValueError(f"Unknown coupling type '{typ}'. Use 'sym' or 'asym'.")

        # Onsite term: Mass term + Coupling
        T0 = u * np.kron(sz, s0) + coup

        # Hopping terms (BHZ model style)
        Tx_plus = 0.5 * np.kron(sz, s0) - 0.5j * np.kron(sx, sz)
        Ty_plus = 0.5 * np.kron(sz, s0) - 0.5j * np.kron(sy, s0)

        # --- Magnetic Block (2x2) ---
        # Zeeman splitting + Chemical potential
        B = b[0] * sx + b[1] * sy + b[2] * sz + mu * s0
        B_x = t_x * s0
        B_y = t_y * s0

        # --- Interlayer Coupling (2x2) ---
        TI_coup = c_inter * s0

        # --- Assemble Full 6x6 Matrices ---
        U = np.zeros((self.CELL_SIZE, self.CELL_SIZE), dtype=complex)
        Tx = np.zeros_like(U)
        Ty = np.zeros_like(U)

        # 1. Onsite Matrix (U)
        U[:self.TI_SIZE, :self.TI_SIZE] = T0       # TI Diagonal
        U[self.TI_SIZE:, self.TI_SIZE:] = B        # Mag Diagonal
        
        # Interlayer Coupling (off-diagonal blocks)
        # Note: Mapping 2x2 identity into specific sub-blocks of the 4x2 / 2x4 space
        # This implementation couples both s and p orbitals to the magnetic layer
        U[self.TI_SIZE:, 0:2] = TI_coup
        U[0:2, self.TI_SIZE:] = TI_coup
        U[self.TI_SIZE:, 2:4] = TI_coup
        U[2:4, self.TI_SIZE:] = TI_coup

        # 2. Hopping Matrices (Tx, Ty)
        Tx[:self.TI_SIZE, :self.TI_SIZE] = Tx_plus
        Tx[self.TI_SIZE:, self.TI_SIZE:] = B_x

        Ty[:self.TI_SIZE, :self.TI_SIZE] = Ty_plus
        Ty[self.TI_SIZE:, self.TI_SIZE:] = B_y

        # Cache and return
        self.U, self.Tx, self.Ty = U, Tx, Ty
        return U, Tx, Ty

    # -------------------------------------------------------------------------
    # Internal Helpers (sisl specific)
    # -------------------------------------------------------------------------
    @staticmethod
    def _spin_order(mat_2x2: np.ndarray) -> np.ndarray:
        """
        Flattens a 2x2 complex matrix into the specific 8-element real vector format
        required by sisl's spin-orbit sparse matrix storage.
        
        The permutation [0, 3, 1, 5, 4, 7, 2, 6] is specific to the internal 
        storage layout of the specific sisl version used.
        """
        # [real_00, real_01, real_10, real_11, imag_00, imag_01, imag_10, imag_11]
        flat = np.append(mat_2x2.real.flatten(), mat_2x2.imag.flatten())
        
        # Reorder to match sisl's packed spin format
        order = [0, 3, 1, 5, 4, 7, 2, 6]
        return flat[order]

    @staticmethod
    def _spin_order_inverse(vec: np.ndarray) -> np.ndarray:
        """
        Reconstructs a complex 2x2 matrix from the sisl-packed 8-element real vector.
        Inverse operation of `_spin_order`.
        """
        order = np.array([0, 3, 1, 5, 4, 7, 2, 6])
        v = np.asarray(vec)
        
        if v.size != 8:
            raise ValueError(f"Expected input vector of length 8, got {v.size}")

        # Restore original order [re00, re01..., im00, im01...]
        flat = np.empty(8, dtype=float)
        flat[order] = v
        
        real = flat[:4].reshape(2, 2)
        imag = flat[4:].reshape(2, 2)
        return real + 1j * imag

    def _place_block(self, H: si.Hamiltonian, row_orbs, col_orbs, M: np.ndarray) -> None:
        """
        Helper to place a 4x4 matrix M into the Hamiltonian using 2x2 sub-blocks.
        M is split into four 2x2 matrices, each processed by `_spin_order`.
        """
        # sisl requires setting data using the spin-stride (S_idx)
        H[row_orbs[0], col_orbs[0], :H.S_idx] = self._spin_order(M[:2, :2])
        H[row_orbs[0], col_orbs[1], :H.S_idx] = self._spin_order(M[:2, 2:4])
        H[row_orbs[1], col_orbs[0], :H.S_idx] = self._spin_order(M[2:4, :2])
        H[row_orbs[1], col_orbs[1], :H.S_idx] = self._spin_order(M[2:4, 2:4])

    def _generate_pair(
        self,
        H: si.Hamiltonian,
        site: int,
        neighbor: int,
        U: np.ndarray,
        Tx: np.ndarray,
        Ty: np.ndarray,
    ) -> None:
        """
        Populate Hamiltonian elements for a specific (atom, neighbor) pair.
        """
        # Direction vectors for comparison
        x_vec = np.array([self.a, 0, 0])
        y_vec = np.array([0, self.a, 0])
        
        # Vector pointing from site to neighbor
        dir_vec = self.geom.Rij(site, neighbor)

        # ---------------------------------------------------------------------
        # Case A: Magnetic Atom
        # ---------------------------------------------------------------------
        if self.geom.atoms[site] == self.Magnetic:
            # 1. Onsite Term
            # Note: U[4:, 4:] extracts the 2x2 Mag block from the 6x6 U
            H[self.geom.a2o(site), self.geom.a2o(site), :H.S_idx] = self._spin_order(U[4:, 4:])
            H[self.geom.a2o(site), self.geom.a2o(site), H.S_idx] = 1 # Set overlap to 1

            # 2. TI-Mag Coupling (Onsite)
            # Find the TI orbitals associated with the atom at (site - 1)
            # Assumes TI is always the atom immediately preceding the Mag atom
            ti_orbs = self.geom.a2o(site - 1, all=True)
            coupling_block = U[:2, 4:6] # 2x2 coupling block

            # Set Hermitian coupling terms
            H[self.geom.a2o(site), ti_orbs, :H.S_idx] = self._spin_order(coupling_block)
            H[ti_orbs[0], self.geom.a2o(site), :H.S_idx] = self._spin_order(coupling_block)
            H[ti_orbs[1], self.geom.a2o(site), :H.S_idx] = self._spin_order(coupling_block)

            # 3. Hopping Terms (Magnetic Layer)
            mag_hop_x = Tx[4:, 4:]
            mag_hop_y = Ty[4:, 4:]

            if np.allclose(dir_vec, x_vec):
                H[self.geom.a2o(site), self.geom.a2o(neighbor), :H.S_idx] = self._spin_order(mag_hop_x)
            elif np.allclose(dir_vec, -x_vec):
                H[self.geom.a2o(site), self.geom.a2o(neighbor), :H.S_idx] = self._spin_order(mag_hop_x.conj().T)
            elif np.allclose(dir_vec, y_vec):
                H[self.geom.a2o(site), self.geom.a2o(neighbor), :H.S_idx] = self._spin_order(mag_hop_y)
            elif np.allclose(dir_vec, -y_vec):
                H[self.geom.a2o(site), self.geom.a2o(neighbor), :H.S_idx] = self._spin_order(mag_hop_y.conj().T)

        # ---------------------------------------------------------------------
        # Case B: TI Atom
        # ---------------------------------------------------------------------
        else:
            ti_orbs = self.geom.a2o(site, all=True)
            nei_orbs = self.geom.a2o(neighbor, all=True)

            # 1. Onsite Term
            self._place_block(H, ti_orbs, ti_orbs, U)
            # Set overlaps
            H[ti_orbs[0], ti_orbs[0], H.S_idx] = 1
            H[ti_orbs[1], ti_orbs[1], H.S_idx] = 1

            # 2. Hopping Terms
            if np.allclose(dir_vec, x_vec):
                self._place_block(H, ti_orbs, nei_orbs, Tx)
            elif np.allclose(dir_vec, -x_vec):
                self._place_block(H, ti_orbs, nei_orbs, Tx.conj().T)
            elif np.allclose(dir_vec, y_vec):
                self._place_block(H, ti_orbs, nei_orbs, Ty)
            elif np.allclose(dir_vec, -y_vec):
                self._place_block(H, ti_orbs, nei_orbs, Ty.conj().T)

    # -------------------------------------------------------------------------
    # Hamiltonian Assembly
    # -------------------------------------------------------------------------
    def build_hamiltonian(
        self,
        U: Optional[np.ndarray] = None,
        Tx: Optional[np.ndarray] = None,
        Ty: Optional[np.ndarray] = None,
    ) -> si.Hamiltonian:
        """
        Assembles the sisl Hamiltonian object.

        This iterates over the geometry and fills the sparse matrix with the 
        tight-binding parameters defined in the unit cell.

        Parameters
        ----------
        U, Tx, Ty : np.ndarray, optional
            Explicit matrices to use. If None, uses cached values from `build_unit_cell`.

        Returns
        -------
        sisl.Hamiltonian
            The populated Hamiltonian.
        """
        if self.geom is None:
            raise ValueError("Geometry not built. Call build_geometry() first.")

        # Fallback to cached matrices
        U = U if U is not None else getattr(self, "U", None)
        Tx = Tx if Tx is not None else getattr(self, "Tx", None)
        Ty = Ty if Ty is not None else getattr(self, "Ty", None)

        if any(m is None for m in (U, Tx, Ty)):
            raise ValueError("Matrices missing. Call build_unit_cell() or provide arguments.")

        # Initialize Spin-Orbit Hamiltonian
        # orthogonal=False implies we handle overlaps (though here overlaps are effectively identity)
        H = si.Hamiltonian(self.geom, spin="spin-orbit", orthogonal=False)

        # Iterate over atoms and neighbors to fill the matrix
        # R=[0.1, self.a] ensures we only catch nearest neighbors
        for ia in self.geom.iter():
            _, neighbors = self.geom.close(ia, R=[0.1, self.a])
            for nb in neighbors:
                self._generate_pair(H, ia, nb, U, Tx, Ty)

        self.H = H
        return H

    # -------------------------------------------------------------------------
    # Analysis: Density Matrix & Magnetism
    # -------------------------------------------------------------------------
    def _fermi(self, E: np.ndarray, kT: float, mu: float) -> np.ndarray:
        """Fermi-Dirac distribution function."""
        if kT == 0.0:
            return 1.0 - np.heaviside(E - mu, 0.5)
        return si.fermi_dirac(E, kT=kT, mu=mu)

    def build_density_matrix(self, kT: float = 0.0, mu: float = 0.0) -> si.DensityMatrix:
        """
        Computes the density matrix from the Hamiltonian eigenstates.

        Parameters
        ----------
        kT : float
            Thermal energy (temperature * k_B) in eV.
        mu : float
            Chemical potential in eV.

        Returns
        -------
        sisl.DensityMatrix
            The computed density matrix.
        """
        if self.geom is None or self.H is None:
            raise ValueError("Geometry or Hamiltonian not built.")

        # Diagonalize Hamiltonian
        es = self.H.eigenstate()
        
        # Calculate occupations
        f_dist = lambda E: self._fermi(E, kT=kT, mu=mu)
        occ = es.occupation(distribution=f_dist)

        # Construct Density Matrix (sum over occupied states)
        # DM_ij = sum_n f(E_n) * psi_n(i) * conj(psi_n(j))
        dm_full = np.zeros(es.state.shape, dtype=complex)
        for i in range(es.eig.shape[0]):
            if occ[i] > 1e-12: # optimization: skip empty states
                dm_full += occ[i] * np.outer(es.state[:, i], np.conj(es.state[:, i]))

        # Map back to sisl sparse format
        DM = si.DensityMatrix(self.geom, spin="spin-orbit")
        nx, ny = dm_full.shape
        Nx, Ny, _ = DM.shape

        for i in range(Nx):
            for j in range(Ny):
                # Map sparse indices to full matrix indices
                id_x = (2 * i) % nx
                id_y = (2 * j) % ny
                # Extract 2x2 spin block and pack it
                block = dm_full[id_x : id_x + 2, id_y : id_y + 2]
                DM[i, j] = self._spin_order(block)

        self.DM = DM
        return DM
    
    def magnetic_moment(self) -> np.ndarray:
        """
        Calculates the total magnetic moment vector of the system.

        Returns
        -------
        np.ndarray
            The magnetic moment vector [Mx, My, Mz] in Bohr magnetons.
        """
        if self.DM is None:
            raise ValueError("Density matrix not built. Call build_density_matrix() first.")

        # Create a copy to manipulate connection info for trace calculation
        dm_copy = self.DM.copy()
        dm_copy.set_nsc([1] * 3) # Isolate unit cell

        Sx_val = 0.0
        Sy_val = 0.0
        Sz_val = 0.0

        # Trace over all orbitals
        for p in range(dm_copy.no):
            # Extract local 2x2 density matrix for orbital p
            rho_pp = self._spin_order_inverse(dm_copy[p, p])
            
            # Expectation values of Pauli matrices
            Sx_val += np.trace(self._SX @ rho_pp).real
            Sy_val += np.trace(self._SY @ rho_pp).real
            Sz_val += np.trace(self._SZ @ rho_pp).real
            
        # Magnetic moment = - Expectation value of spin
        return -np.array([Sx_val, Sy_val, Sz_val])
    
    def automate(
        self, 
        params: Optional[TopologicalInsulatorParams] = None, 
        Nx: int = 4, 
        Ny: int = 4, 
        a: float = 1.0, 
        z: float = 10.0, 
        abc: Optional[List[int]] = None, 
        kT: float = 0.01, 
        mu_chem: float = 0.0, 
        **kwargs
    ) -> Tuple[si.Hamiltonian, si.DensityMatrix]:
        """
        Executes the full simulation pipeline: Geometry -> Unit Cell -> Hamiltonian -> DM.
        
        Parameters
        ----------
        params : TopologicalInsulatorParams, optional
            Model parameters.
        Nx, Ny, a, z : float/int
            Geometry parameters.
        abc : list, optional
            Boundary conditions.
        kT, mu_chem : float
            Thermodynamic parameters for the density matrix.
        **kwargs :
            Direct overrides for `params` (e.g., b=(0,0,1)).

        Returns
        -------
        (Hamiltonian, DensityMatrix)
        """
        # 1. Setup Parameters
        if params is None:
            params = TopologicalInsulatorParams(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(params, k):
                    setattr(params, k, v)

        # 2. Build Geometry
        self.build_geometry(Nx=Nx, Ny=Ny, a=a, z=z, abc=abc)

        # 3. Build Unit Cell Matrices
        self.build_unit_cell(params)

        # 4. Build Hamiltonian & DM
        H = self.build_hamiltonian()
        DM = self.build_density_matrix(kT=kT, mu=mu_chem)
        
        return H, DM


@dataclass
class MagneticParams:
    """
    Parameters for a standalone 2D magnetic layer.
    """
    b: tuple = (1, 0, 0)  # Magnetic exchange field (Bx, By, Bz)
    t_x: float = -0.1     # Hopping in x
    t_y: float = -0.1     # Hopping in y
    mu: float = 0.0       # Chemical potential

class MagneticModel:
    """
    A standalone 2D magnetic lattice model using sisl.
    Basis: 1 atom per cell, 2 spin orbitals.
    """
    CELL_SIZE = 2

    _SX = np.array([[0, 1], [1, 0]], dtype=complex)
    _SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _SZ = np.array([[1, 0], [0, -1]], dtype=complex)
    _S0 = np.eye(2, dtype=complex)

    def __init__(self) -> None:
        self.geom = None
        self.H = None

    def build_geometry(
        self,
        Nx: int,
        Ny: int,
        a: float,
        abc: Optional[List[int]] = None,
    ) -> si.Geometry:
        """
        Builds geometry using the user-defined boundary condition mapping.
        """
        self.Nx, self.Ny, self.a = Nx, Ny, a

        if abc is None:
            abc = [5, 2, 5]  # Default: periodic in y, finite in x/z

        # Your original mapping logic: (2 → 3, else → 1)
        nsc = [3 if code == 2 else 1 for code in abc]

        # Single magnetic atom
        mag_atom = si.Atom(1)
        
        # Lattice with your calculated nsc
        lattice = si.Lattice([a, a, 10.0], nsc=nsc)
        lattice.set_boundary_condition(a=abc[0], b=abc[1], c=abc[2])

        # Single atom basis at center of cell
        geom = si.Geometry([[a/2, a/2, 0]], atoms=[mag_atom], lattice=lattice)
        self.geom = geom.tile(Nx, 0).tile(Ny, 1)
        
        return self.geom

    def build_unit_cell(self, params: MagneticParams):
        """Constructs 2x2 onsite and hopping matrices."""
        sx, sy, sz, s0 = self._SX, self._SY, self._SZ, self._S0
        
        # Onsite: Exchange field + Chemical potential
        self.U = params.b[0]*sx + params.b[1]*sy + params.b[2]*sz + params.mu*s0
        
        # Hopping: Simple scalar hopping (conserves spin)
        self.Tx = params.t_x * s0
        self.Ty = params.t_y * s0
        return self.U, self.Tx, self.Ty

    @staticmethod
    def _spin_order(mat_2x2: np.ndarray) -> np.ndarray:
        """Sisl spin-orbit vector packing."""
        order = [0, 3, 1, 5, 4, 7, 2, 6] 
        flat = np.append(mat_2x2.real.flatten(), mat_2x2.imag.flatten())
        return flat[order]
    
    @staticmethod
    def _spin_order_inverse(vec: np.ndarray) -> np.ndarray:
        """
        Inverse of _spin_order: reconstruct a 2x2 complex matrix from the 8-element
        real spin-vector produced by _spin_order.

        Parameters
        ----------
        vec : array-like, shape (8,)
            Real vector = [real.flatten(), imag.flatten()] reordered by `order`.

        Returns
        -------
        mat_2x2 : np.ndarray, shape (2,2), dtype=complex
            Reconstructed complex 2x2 matrix.
        """
        order = np.array([0, 3, 1, 5, 4, 7, 2, 6])
        v = np.asarray(vec)
        if v.size != order.size:
            raise ValueError(f"expected input vector of length {order.size}, got {v.size}")
        # recover the original concatenated real+imag flat array
        flat = np.empty(order.size, dtype=float)
        flat[order] = v
        real = flat[:4].reshape(2, 2)
        imag = flat[4:].reshape(2, 2)
        return real + 1j * imag

    def build_hamiltonian(self) -> si.Hamiltonian:
        if self.geom is None: raise ValueError("Build geometry first.")
        
        # Initialize SO Hamiltonian
        H = si.Hamiltonian(self.geom, spin="spin-orbit",orthogonal = False)


        for ia in self.geom.iter():
            # Onsite term
            H[ia, ia, :H.S_idx] = self._spin_order(self.U)
            H[ia, ia ,H.S_idx] = 1

            # Neighbors for hopping
            _, neighbors = self.geom.close(ia, R=[0.1, self.a + 0.1])
            for nb in neighbors:
                dist = self.geom.Rij(ia, nb)
                
                if np.allclose(dist, [self.a, 0, 0]):
                    H[ia, nb, :H.S_idx] = self._spin_order(self.Tx)
                elif np.allclose(dist, [-self.a, 0, 0]):
                    H[ia, nb, :H.S_idx] = self._spin_order(self.Tx.conj().T)
                elif np.allclose(dist, [0, self.a, 0]):
                    H[ia, nb, :H.S_idx] = self._spin_order(self.Ty)
                elif np.allclose(dist, [0, -self.a, 0]):
                    H[ia, nb, :H.S_idx] = self._spin_order(self.Ty.conj().T)
        
        self.H = H
        return H
    
        # ---------- Helper for distrobution ----------
    def _fermi(self, E, kT, mu):
        if kT == 0.0:
            return 1-np.heaviside(E,0.5)
        else:
            return si.fermi_dirac(E, kT=kT, mu=mu)
    
    def build_density_matrix(self, kT: float = 0.0, mu: float = 0.0) -> si.DensityMatrix:
        """
        Construct the density matrix for the magnetic layer from eigenstates.
        """
        if self.H is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian() first.")

        # 1. Get Eigenstates
        es = self.H.eigenstate()
        
        # 2. Define Fermi-Dirac distribution
        # si.fermi_dirac is efficient, but we handle the T=0 case for stability
        f = lambda E: self._fermi(E, kT=kT, mu=mu)
        
        # 3. Calculate occupations
        occ = es.occupation(distribution=f)
        
        # 4. Construct the dense DM from eigenstates: \rho = \sum n_i |\psi_i><\psi_i|
        # es.state has shape (total_orbitals, total_states)
        dm_dense = np.zeros(es.state.shape, dtype=complex)
        for i in range(es.eig.shape[0]):
            dm_dense += occ[i] * np.outer(es.state[:, i], np.conj(es.state[:, i]))

        # 5. Initialize sisl DensityMatrix object
        DM = si.DensityMatrix(self.geom, spin="spin-orbit")

        # 6. Map the dense matrix back into sisl's sparse format
        # Each atom has 1 sisl-orbital, but because it is 'spin-orbit', 
        # that orbital contains the 2x2 spin information.
        nx, ny = dm_dense.shape
        Nx, Ny, _ = DM.shape

        for i in range(Nx):
            for j in range(Ny):
                id_x = (2 * i) % nx
                id_y = (2 * j) % ny
                DM[i, j] = self._spin_order(dm_dense[id_x:id_x + 2, id_y:id_y + 2])

        self.DM = DM
        return DM

    def magnetic_moment(self, kT: float = 0.01, mu: float = 0.0):
        if not hasattr(self, "DM") or self.DM is None:
            raise ValueError("Density matrix not built. Call build_density_matrix() first.")

        copy = self.DM.copy()
        copy.set_nsc([1] * 3)

        Sx_val = 0.0
        Sy_val = 0.0
        Sz_val = 0.0

        for p in range(copy.no):
            rho_pp_block = self._spin_order_inverse(copy[p, p])
            Sx_val += np.trace( self._SX @ rho_pp_block).real
            Sy_val += np.trace( self._SY @ rho_pp_block).real
            Sz_val += np.trace( self._SZ @ rho_pp_block).real
            
        mu_x = -Sx_val
        mu_y = -Sy_val
        mu_z = -Sz_val
        return - np.array([mu_x, mu_y, mu_z])
    
    def automate(self, params=None, Nx=10, Ny=1, a=1.0, abc=None, kT=0.0, mu=0.0, **kwargs):
        """
        Executes the full pipeline: Geometry -> Unit Cell -> Hamiltonian -> Density Matrix.
        
        Parameters
        ----------
        params : MagneticParams, optional
            Dataclass with model parameters.
        Nx, Ny, a : float
            Geometry and lattice parameters.
        abc : list, optional
            Boundary conditions. Defaults to [5, 2, 5].
        kT, mu : float
            Thermal energy and chemical potential for the Density Matrix.
        **kwargs :
            Direct overrides for params ( e.g., b=(0,0,1) ).
        """
        # 1. Setup Parameters
        if params is None:
            params = MagneticParams(**kwargs)
        else:
            # Override specific dataclass fields if kwargs are provided
            for k, v in kwargs.items():
                if hasattr(params, k):
                    setattr(params, k, v)

        self.build_geometry(Nx=Nx, Ny=Ny, a=a, abc=abc)

        # 3. Build Unit Cell Matrices
        self.build_unit_cell(params)

        # 4. Build Density Matrix
        return self.build_hamiltonian(), self.build_density_matrix(kT=kT, mu=mu)
    

def build_lattice_pairs(Nx, Ny, radius, atoms_per_cell=1):
    """
    Robust pair builder that handles multi-atom bases.
    """
    all_pairs = []
    r_int = int(np.ceil(radius))
    r_sq_limit = radius**2 + 1e-6

    # 1. Loop over every unit cell in the grid
    for cx in range(Nx):
        for cy in range(Ny):
            
            # 2. Loop over every atom inside the current unit cell (ai_local)
            for ai_local in range(atoms_per_cell):
                # Calculate the global index of the current atom
                ai = (cx + cy * Nx) * atoms_per_cell + ai_local
                
                # 3. Look for neighbors in nearby unit cells (dx, dy)
                for dx in range(-r_int, r_int + 1):
                    for dy in range(-r_int, r_int + 1):
                        
                        # Raw neighbor cell coordinates
                        nx_cell = cx + dx
                        ny_cell = cy + dy
                        
                        # --- Boundary Conditions ---
                        # X: Finite
                        if nx_cell < 0 or nx_cell >= Nx:
                            continue 

                        # Y: Periodic
                        R_vec = [0, 0, 0]
                        if ny_cell < 0 or ny_cell >= Ny:
                            R_vec[1] = ny_cell // Ny
                            ny_cell = ny_cell % Ny
                        
                        # 4. Loop over every atom in the neighbor unit cell (aj_local)
                        for aj_local in range(atoms_per_cell):
                            # Calculate global index of the neighbor atom
                            aj = (nx_cell + ny_cell * Nx) * atoms_per_cell + aj_local
                            
                            # Skip if it's the exact same atom (same cell AND same local index)
                            if dx == 0 and dy == 0 and ai_local == aj_local:
                                continue
                            
                            # Distance check (Distance between cells)
                            if dx*dx + dy*dy > r_sq_limit:
                                continue

                            all_pairs.append({
                                'ai': ai,
                                'aj': aj,
                                'Ruc': R_vec
                            })

    return all_pairs

def build_lattice_pairs_periodic(Nx, Ny, radius, atom_indices=None):
    """
    Iterates over a list of atoms and collects all neighbor pairs with 
    periodic boundary conditions in BOTH X and Y directions.
    """
    if atom_indices is None:
        atom_indices = range(Nx * Ny)
    
    all_pairs = []
    r_int = int(np.ceil(radius))
    r_sq_limit = radius**2 + 1e-6

    for ai in atom_indices:
        cx = ai % Nx
        cy = ai // Nx
        
        for dx in range(-r_int, r_int + 1):
            for dy in range(-r_int, r_int + 1):
                if dx == 0 and dy == 0:
                    continue
                
                if dx*dx + dy*dy > r_sq_limit:
                    continue
                
                # Raw neighbor coordinates before wrapping
                nx_raw = cx + dx
                ny_raw = cy + dy
                
                # R_vec tracks how many unit cell lengths we jumped
                R_vec = [0, 0, 0]
                
                # --- Periodic Boundary Conditions ---
                
                # Wrap X
                R_vec[0] = nx_raw // Nx  # Track wraparound for Hamiltonian phase
                nx = nx_raw % Nx         # Map back to [0, Nx-1]
                
                # Wrap Y
                R_vec[1] = ny_raw // Ny
                ny = ny_raw % Ny         # Map back to [0, Ny-1]
                
                # Convert wrapped (nx, ny) back to global index 'aj'
                aj = nx + ny * Nx
                
                all_pairs.append({
                    'ai': ai,
                    'aj': aj,
                    'Ruc': R_vec
                })

    return all_pairs

from scipy.optimize import brentq
from scipy.special import expit # Sigmoid function 1/(1+e^-x), numerically stable

def calculate_chemical_potential(H: si.Hamiltonian, filling_per_cell: float, kT: float = 0.01) -> float:
    """
    Calculates the chemical potential (mu) for a specific filling.
    
    Parameters
    ----------
    H : sisl.Hamiltonian
        The Hamiltonian (can be a supercell).
    filling_per_cell : float
        Target number of electrons **per unit cell**. 
        (e.g., 3.0 for half-filling of a 6-orbital basis).
    kT : float
        Thermal energy in eV.
        
    Returns
    -------
    mu : float
        The calculated chemical potential.
    """
    # 1. Get eigenvalues
    # Since you tiled the geometry (Nx, Ny), the Gamma-point (default) 
    # of the supercell effectively samples the k-space of the primitive cell.
    eigenvalues = H.eigenstate().eig 
    
    # Calculate the total target electrons for the WHOLE supercell
    # We can infer the number of unit cells from the geometry
    n_cells = H.geometry.na // 2  # Assuming 2 atoms per primitive cell (TI + Mag)
    # OR safer: calculate ratio of supercell atoms to primitive atoms if known
    # If you are unsure, just pass 'total_electrons' as an argument instead.
    
    # Simpler approach: Assume user provides total electrons or calculate density
    target_total = filling_per_cell * (H.geometry.na / 2) 
    # Note: Division by 2 assumes your primitive basis has exactly 2 atoms (TI + Mag).
    # If your basis changes, adjust this logic.

    # 2. Define the counting function
    def count_difference(mu_guess):
        if kT < 1e-6: # Treat very low T as 0
            count = np.sum(eigenvalues <= mu_guess)
        else:
            # ROBUST CALCULATION:
            # expit(x) = 1 / (1 + exp(-x))
            # We want 1 / (exp((E-mu)/kT) + 1)
            # This is equivalent to expit(-(E-mu)/kT)
            x = -(eigenvalues - mu_guess) / kT
            occ = expit(x)
            count = np.sum(occ)
            
        return count - target_total

    # 3. Define search bounds (with padding to ensure root is bracketed)
    min_E = np.min(eigenvalues) - 0.5
    max_E = np.max(eigenvalues) + 0.5

    # 4. Solve
    try:
        mu_solution = brentq(count_difference, min_E, max_E)
    except ValueError:
        # Fallback if bounds are too tight (rare)
        mu_solution = brentq(count_difference, min_E - 5.0, max_E + 5.0)
    
    return mu_solution


def generate_custom_matrices(
    u: float = -0.28,         # TI mass (bandgap parameter)
    t_ti: float = 0.5,        # TI normal hopping (effective mass)
    v_ti: float = 0.5,        # TI spin-orbit hopping (Dirac velocity)
    c_intra: float = 0.0,     # TI internal orbital coupling
    typ: str = "asym",        # Symmetry of TI coupling
    b: tuple = (0.0, 0.0, 0.1), # Magnetic exchange vector (bx, by, bz)
    t_x: float = -1.0,        # Magnetic layer x-hopping
    t_y: float = -1.0,        # Magnetic layer y-hopping
    c_inter: float = 0.1,     # Interlayer coupling
    mu: float = 0.0           # Magnetic layer chemical potential
):
    """
    Generates customized 6x6 tight-binding matrices (U, Tx, Ty) for a 
    Topological Insulator + Ferromagnet heterostructure.
    """
    # 1. Pauli Matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    s0 = np.eye(2, dtype=complex)

    # 2. TI Block (4x4)
    # Intra-orbital coupling
    if typ == "sym":
        coup = np.kron(sx, c_intra * sx)
    elif typ == "asym":
        coup = np.kron(sy, c_intra * sx)
    else:
        raise ValueError("typ must be 'sym' or 'asym'")

    # Onsite (u)
    T0 = u * np.kron(sz, s0) + coup

    # Hopping (t_ti for curvature, v_ti for Dirac velocity)
    Tx_plus = t_ti * np.kron(sz, s0) - (v_ti * 1j) * np.kron(sx, sz)
    Ty_plus = t_ti * np.kron(sz, s0) - (v_ti * 1j) * np.kron(sy, s0)

    # 3. Magnetic Block (2x2)
    B = b[0] * sx + b[1] * sy + b[2] * sz + mu * s0
    B_x = t_x * s0
    B_y = t_y * s0

    # 4. Interlayer Coupling
    TI_coup = c_inter * s0

    # 5. Assemble 6x6 Matrices
    U = np.zeros((6, 6), dtype=complex)
    Tx = np.zeros_like(U)
    Ty = np.zeros_like(U)

    # Fill U (Onsite)
    U[:4, :4] = T0
    U[4:, 4:] = B
    U[4:, 0:2] = TI_coup
    U[0:2, 4:] = TI_coup
    U[4:, 2:4] = TI_coup
    U[2:4, 4:] = TI_coup

    # Fill Tx (X-Hopping)
    Tx[:4, :4] = Tx_plus
    Tx[4:, 4:] = B_x

    # Fill Ty (Y-Hopping)
    Ty[:4, :4] = Ty_plus
    Ty[4:, 4:] = B_y

    return U, Tx, Ty