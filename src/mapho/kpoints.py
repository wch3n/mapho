"""Geometry-based, Gamma-centred meshes; convergence requires DFT checks."""

from importlib.metadata import version

import numpy as np


def mesh_from_spacing(atoms, spacing=0.4):
    """Return a pymatgen mesh with |b_i|/n_i <= spacing_i (inverse Angstrom).

    Reciprocal vectors include 2*pi. A scalar or three directional spacings
    are accepted. The reciprocal metric is used even for skewed cells; these
    are spacings along reciprocal basis vectors, not a covering-radius bound.
    Pymatgen supplies the integer mesh; ASE writes the actual QE input syntax.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.inputs import Kpoints

    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Reciprocal spacings must be finite and positive")
    if not np.all(atoms.pbc) or abs(np.linalg.det(atoms.cell)) < 1e-10:
        raise ValueError("A non-singular fully periodic cell is required")
    lattice = Lattice(np.asarray(atoms.cell))
    structure = Structure(
        lattice, atoms.get_chemical_symbols(), atoms.get_scaled_positions()
    )
    reciprocal = np.array(lattice.reciprocal_lattice.abc)
    lengths = np.array(lattice.abc)
    # Pymatgen divides length_densities by |a_i| before rounding up.
    # The metric correction gives ceil(|b_i|/spacing_i), also for skew cells.
    densities = reciprocal * lengths / spacing
    kp = Kpoints.automatic_density_by_lengths(structure, densities, force_gamma=True)
    mesh = tuple(int(n) for n in kp.kpts[0])
    return dict(
        mesh=list(mesh),
        offset=[0, 0, 0],
        centering="Gamma",
        requested_spacing_inv_angstrom=spacing.tolist(),
        actual_spacing_inv_angstrom=(reciprocal / mesh).tolist(),
        reciprocal_lengths_inv_angstrom=reciprocal.tolist(),
        pymatgen_version=version("pymatgen"),
        method="automatic_density_by_lengths with reciprocal-metric correction",
    )
