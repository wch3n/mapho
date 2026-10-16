# Quantum ESPRESSO scalar-potential backend

This backend exports real PBE/norm-conserving DFT data and solves a supplied
Hartree–XC potential with QEpy/Quantum ESPRESSO. The kinetic, ionic local, and
nonlocal projector operators remain in QE. No dense plane-wave matrix is built.

The first milestone is a numerical interface test: reproduce a converged
calculation after exporting and reinserting its electronic potential, using
a fresh process, atomic initial density, and random starting wavefunctions.
A trained MACE/FNO grid-potential pilot now builds on this interface; see
[field_learning.md](field_learning.md). The orbital learner remains a
separate baseline.

## Isolated runtime

The serial wheel was tested on Linux x86-64 with Python 3.12.12. Keep this
environment separate from an existing MACE installation:

```bash
cd /path/to/mapho
python3 -m venv .venv-qe
env -u PYTHONPATH .venv-qe/bin/python -m pip install -r requirements-qe.txt
env -u PYTHONPATH .venv-qe/bin/python -m pip install --no-deps -e .
```

The `--no-deps` installation registers this project's Python sources without
installing its orbital-training dependencies (PyTorch/e3nn) into the QE
runtime. The QE commands import NumPy, ASE and QEpy only. Run the normal test
suite in the existing learning environment, which includes PyTorch.

QEpy 7.2.0's published wheel requires `f90wrap==0.2.14` for its wrapped Fortran
handles. With f90wrap 0.3.0, SCF runs but reading the density/potential arrays
fails. The backend checks these versions before starting a calculation.
`requirements-qe.txt` also pins NumPy 1.26.4.

## Reproduce the silicon test

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-qe/bin/python -m mapho.qe silicon --output runs/si_check
```

Output directories must be new. By default this downloads the unmodified
SG15 Si ONCV PBE 1.2 UPF file and verifies its SHA-256:

```text
73d21061e1050ef8134ba50a20e61b9d0bcef98d0a9e0a110bbd076e47282eee
```

The download URL, original license text and file checksum are retained. A
local UPF can instead be supplied with `--pseudo /path/to/Si.upf`. The file
must be UPF v2, scalar/nonrelativistic, norm-conserving, and PBE.

The example uses two-atom diamond Si at 5.43 angstrom, 40 Ry wavefunction
cutoff, 160 Ry density cutoff, a 2x2x2 k mesh, eight bands, and fixed
spin-degenerate occupations. These are interface-test settings, not a
converged production band-gap benchmark.

The command runs:

1. A converged SCF reference.
2. A frozen-potential replay in a fresh process.
3. The same potential plus 0.2 eV: all eigenvalues must shift by 0.2 eV,
   while the density stays unchanged.
4. A 0.5 eV cosine perturbation along the first fractional lattice direction:
   the density must change, detecting a solver that ignores the supplied field.

The replay explicitly sets `exttype=("hartree", "xc")`, rebuilds the applied
potential, and verifies that it equals the supplied Hxc field plus QE's ionic
local potential. It then diagonalizes at a fixed potential and reconstructs
the density. It never calls an SCF update during replay. The default external
potential mode would instead add the field to the existing Hxc contribution.

Cold-start diagonalization may need multiple Davidson passes; eigenvalue
stability is checked with a bounded loop. No converged reference density or
wavefunctions are supplied to the replay.

Results from this workspace:

| Check | Diamond Si | Distorted Si |
|---|---:|---:|
| Maximum eigenvalue error (eV) | 4.17e-6 | 6.53e-7 |
| Integrated absolute density error / electron count | 1.43e-7 | 6.33e-8 |
| Constant-shift eigenvalue error (eV) | 2.98e-12 | 8.98e-13 |
| Constant-shift relative density change | 6.96e-11 | 9.59e-11 |
| Cosine perturbation relative density change | 0.0326 | 0.0351 |

The distorted cell applies the row-vector deformation
`[[1.13, .06, 0], [0, .94, .03], [0, 0, 1.03]]` and moves the second atom
by `[.07, -.03, .02]` angstrom. This also exercises unequal FFT grid sizes.
Exact errors may vary with platform. Acceptance tolerances are recorded in
each `validation.json`.

Reports and logs are under `runs/qe_si_roundtrip` and `runs/qe_si_distorted`.
Each worker records versions, generated QE input, solver log, and elapsed time.

## Export another structure

```bash
env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe reference \
  --structure sample.extxyz --pseudo Si=/path/to/Si.upf \
  --ecutwfc 40 --kmesh 2 2 2 --nbnd 8 --output runs/sample
```

Repeat `--pseudo Element=path` for each element. The current backend supports
neutral, fully periodic, nonmagnetic PBE/NC systems with an even electron
count and fixed occupations. Use insulating samples. Smearing for metals,
SOC, charged systems, PAW/ultrasoft, hybrids, DFT+U, and MPI are not exposed.
A separate experimental [HSE exchange-action exporter](exchange.md) supports
the fixed silicon pilot without changing this PBE field contract.
A slab can be represented in a periodic cell with vacuum, but this backend
does not yet expose a slab Coulomb cutoff or dipole correction.

The pseudopotential remains a required input at inference. Every solve
checks its checksum. Stored paths are relative to the reference directory;
copy `pseudos/` along with `fields.npz` to move a reference.

## Field contract

`FieldSample.load("fields.npz")` validates the NPZ without loading pickle:

- Cell vectors are rows, positions/cell are in angstrom, eigenvalues and
  potentials in eV, and densities in electrons per cubic angstrom.
- Each scalar field has shape `(nx, ny, nz)`. Point `(i, j, k)` is at
  `[i/nx, j/ny, k/nz] @ cell`. The origin is zero; periodic endpoints are
  excluded. QE's Fortran storage has the first lattice index varying fastest.
- Fields contain density, ionic local potential, Hartree potential, XC
  potential, initial atomic-superposition density and its Hxc potential.
- k points are fractional reciprocal coordinates, weights sum to one, and
  occupations range from zero to two. Both density integration and weighted
  occupations must reproduce the declared electron count.
- Metadata records functional, spin, geometry/grid conventions, native QE
  potential reference, exact pseudopotential identities, conversion constants,
  cutoffs, k mesh, SCF convergence and software versions.

The candidate learning target is

```text
delta_Vhxc = Vhxc_converged - Vhxc_atomic_reference
```

`sample.residual()` returns its zero-mean field and its separate mean offset.
`target.npz` stores those as `residual_eV` and `mean_offset_eV`. Reconstruct
with `reference_hxc + residual_eV + mean_offset_eV`; dropping the offset shifts
the eigenvalues. These files are companions to `fields.npz`, not standalone
geometry/gauge-free labels. This convention does not automatically align
potentials between different codes or pseudopotentials.

## Solve a predicted potential

Save a full Hxc array on the reference grid as a NumPy `.npy` file in eV:

```bash
env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe solve \
  --reference runs/sample/fields.npz --hxc predicted_hxc_eV.npy \
  --output runs/predicted_solve
```

Omit `--hxc` to replay the reference. `solution.npz` contains eigenvalues,
density, occupations, k weights and the applied local potential. It does not
report a DFT total energy or forces: an arbitrary learned potential needs
additional consistency work before those quantities can be interpreted.

## Bands at new k points

The `bands` command diagonalizes either the converged reference potential or
a supplied prediction at an explicit list of fractional reciprocal k points.
The geometry, pseudopotential, cutoffs and real-space potential grid remain
those of the reference. An `(N, 3)` NumPy file supplies the full point list:

```bash
env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe bands \
  --reference runs/sample/fields.npz --kpoints kpoints.npy \
  --output runs/sample_bands_qe

env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe bands \
  --reference runs/sample/fields.npz --kpoints kpoints.npy \
  --hxc predicted_hxc_eV.npy --output runs/sample_bands_predicted
```

The Python equivalent is `solve(reference, output, hxc=potential,
kpoints=fractional_points)`. An atomic reference still requires an explicit
predicted Hxc. `bands.npz` stores eigenvalues, actual fractional k points,
cell, electron count, potential gauge, the applied potential and convergence
diagnostics. The worker verifies the requested k-point order and stops only
when eigenvalues stabilize. It performs no SCF update and does not integrate
a density from the path; the integration-mesh `solve` behavior is unchanged.

A dense path improves band sampling but does not improve the original SCF
k-mesh convergence. For a distorted structure, labels inherited from a parent
high-symmetry cell must be identified as that convention. Both spectra should
use the same energy zero when assessing absolute prediction errors.

The workspace example in `runs/si_joint_finetune_v1/bands_xgl` evaluates the
first held-out structure on 81 points along the parent diamond X–Gamma–L
path. It checks endpoints against the existing mesh solutions, verifies a
constant-potential shift at three path points, and replays the original
integration-mesh calculation. Both the path generator and source data are
retained with the figure.

## Field-learning experiment

The implemented experiment trains a MACE atom-to-grid decoder, freezes it,
then fits an FNO residual and evaluates both through this solver. It compares
the atomic-superposition, local, and local-plus-FNO potentials on held-out
silicon structures. Commands, model details, and scope are in
[the field-learning guide](field_learning.md).

For a new geometry, `python -m mapho.qe atomic` accepts the same structure,
pseudopotential and solver arguments as `reference`, but exports only
`atomic.npz` without running SCF. This contains the initial density/Hxc,
geometry and operator metadata, without converged labels. The model's
`predict` command accepts it, and `solve` accepts it with an explicit
`--hxc` array. This completes inference without requiring a reference SCF.

## Primary interface references

- [QEpy installation](https://qepy.rutgers.edu/install.html)
- [QEpy driver and potential API](https://qepy.rutgers.edu/source.html)
- [QEpy potential assembly](https://github.com/Quantum-MultiScale/QEpy/blob/dev/src/pw/qepy_v_of_rho_all.f90)
- [UPF specification](https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format)
- [SG15 library](http://www.quantum-simulation.org/potentials/sg15_oncv/)
