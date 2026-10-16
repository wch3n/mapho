# Orbital-block baseline

This independent model maps atomic structures to real-space Hamiltonian
blocks in a complete localized basis. It uses e3nn tensor products and
message passing. It does not use the MACE backbone of the field model.
The demonstration uses analytic synthetic labels.

## Commands

```bash
mapho demo --basis spd --channels 4 --epochs 80 --output runs/spd_example
mapho train --data data --output runs/orbital_model --epochs 200
mapho predict --checkpoint runs/orbital_model/model.pt \
  --structure structure.extxyz --kpoints kpoints.txt --output predicted.npz
```

Use new output directories. Structure files must preserve periodic-boundary
flags. The prediction command rejects elements absent from training.
There are no pretrained material parameters in the repository.

## Basis and geometry

Every atom uses the same complete basis of real spherical harmonics in the
e3nn convention. Presets `s`, `sp`, and `spd` contain 1, 4, and 9 orbitals per
atom. The Python API also accepts repeated shells. Labels in a different
basis require an explicit transformation of order and phase.

Cell rows are lattice vectors in angstroms, positions are Cartesian in
angstroms, and Hamiltonian entries are in eV. A block key is
`[Rx, Ry, Rz, i, j]`, with integer lattice translations and zero-based atom
indices. The corresponding vector is `positions[j] + R @ cell - positions[i]`.
All pairs within the cutoff, including reverse pairs and on-site blocks,
must have labels. Reverse-edge symmetry imposes
`H_ij(R) = H_ji(-R).T`.

The Bloch convention is

$$
H_{ij}(\mathbf k)=\sum_{\mathbf R}H_{ij}(\mathbf R)
\exp(2\pi i\mathbf k\cdot\mathbf R).
$$

The k points are fractional reciprocal coordinates. This cell-periodic gauge
does not include phases from the positions of the orbitals.

## Data contract

`mapho.data.save_sample()` writes NPZ arrays `numbers`, `positions`, `cell`,
`pbc`, `pairs`, `hamiltonian`, and a JSON `metadata` string. Metadata describe
units, basis conventions, cutoff, and spin status. Training requires a common
nonempty `energy_reference` description and a common boolean
`orthogonal_basis`. These declarations record the convention of the labels.
They do not perform energy alignment.

For orthogonal labels, the prediction command also writes eigenvalues.
For nonorthogonal labels, it writes Hamiltonian blocks. Bands then require
independently supplied positive-definite overlap blocks through
`mapho.bands.band_energies(H, graph, kpoints, overlap=S)`. There is no learned
overlap model.

`read_blocks_h5()` and `write_blocks_h5()` transport flat blocks with
`atom_pairs`, `chunk_boundaries`, `chunk_shapes`, and `entries`.
`read_wannier90_hr()` preserves complex entries and applies the supplied
Wigner–Seitz degeneracies. These readers do not determine orbital identities,
phase conventions, or a common gauge across structures. No automatic
Wannier-to-training conversion is provided, and the trainable baseline
accepts real spinless labels only.

Species-dependent basis dimensions, spin, and SOC are outside the present
orbital model. The separate VASP field interface has a different data and
operator contract. See [usage](../usage.md) for the available routes.
