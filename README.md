# MAPHO

**Many-body Potential-based Hamiltonian Operator**

A small Hamiltonian-learning project, with an independent equivariant
orbital-block baseline and a MACE/FNO Quantum ESPRESSO potential learner.
Atomic structures map to real-space orbital blocks, which can be Fourier
transformed and diagonalized. The plane-wave backend exports real DFT fields
and solves supplied potentials using QE's kinetic and nonlocal operators.

The orbital baseline uses custom e3nn message passing and analytic synthetic
demo labels; it does not use the MACE backbone. The field model
directly reuses the installed MACE architecture, adds an atom-to-grid potential
head, and fits an FNO to the remaining error. It has been trained on real
PBE/norm-conserving silicon data with structure-separated evaluation. Optional
joint fine-tuning updates the MACE backbone, field head and FNO together;
a frozen-MACE continuation provides a matched-update control. This
small pilot is **not a state-of-the-art performance or transferability claim**.
An experimental PBE [VASP PAW backend](docs/vasp_paw.md) provides full-operator
export and replay for nonmagnetic (`ISPIN=1`) and collinear magnetic
(`ISPIN=2`) calculations. Native validation is recorded in
`runs/vasp_paw_pbe_v5/validation.json` (Si/LiF) and
`runs/vasp_paw_spin_v1/validation.json` (Fe, including spin reversal).
A joint Si16 PAW training pilot now adds a MACE on-site matrix head and
geometry-only atomic-density references; its fits and native validation are
tracked in `runs/vasp_si16_train_v1`. No ABINIT calculation was run.
The MoS2 pilot now has a [conditioned PAW readout repair](docs/vasp_paw.md#conditioned-paw-readout-repair),
with the original potential model frozen and native checks of the previously
failing spectra. Repaired checkpoints are in `runs/vasp_mos2_paw_conditioned_v3`.

For the plane-wave work, start with [field learning](docs/field_learning.md)
and [the QE backend guide](docs/qe.md).

Experimental [hybrid-exchange pilots](docs/exchange.md) compare MACE,
a pointwise control and MACE–FNO for a constrained exchange correction:
HSE06 in silicon and LiF, and unscreened PBE0 in LiF. Models are trained
separately for each material and hybrid target.
It requires converged PBE orbitals and evaluates exchange separately from
the hybrid local potential; it is not yet a complete hybrid Hamiltonian model.
A separate [VASP PBE0 component test](docs/vasp_pbe0.md) checks the exchange
contribution missing from the present smooth-potential/PAW interface before
extending its training workflow to hybrids.
The [private VASP exchange interface](docs/vasp_exchange.md) now exports full
Fock actions, constructs factors with MAPHO's independent ACE core, and applies
them during frozen PAW diagonalization. Its A100 Si replay passes native-band,
gauge-shift, and reduced-exchange controls. This validates the interface;
it does not yet establish a trained PBE0 model. VASP patchers and Fortran
bridges remain excluded from [public release archives](docs/releasing.md).

Reusable backend code lives in `src/mapho/qe/` and `src/mapho/vasp/`.
VASP argument parsing lives in `src/mapho/commands/vasp/`, and research
protocols live in `src/mapho/experiments/vasp/`. Shared models remain at
`src/mapho/`. See the [package layout](docs/backends.md) for imports, commands,
worker snapshots, and stored data. Use `mapho-vasp --help` for VASP commands.

Python source formatting uses Ruff 0.16.5 with 88-character lines. Install the
optional `dev` dependencies and run `ruff format .` to format or
`ruff format --check .` to verify. The configuration excludes run snapshots,
archives, build output, and virtual environments.

## Run

Python 3.10+, PyTorch, MACE (`mace-torch`), e3nn, ASE, NumPy, and h5py are
required and declared in `pyproject.toml`. The field and PAW models import the
MACE backbone directly. Use a dedicated environment so dependency resolution
does not affect other projects.

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install -e '.[test]'
env -u PYTHONPATH .venv/bin/python -m pytest -q
env -u PYTHONPATH .venv/bin/python -m mapho demo --output runs/example
```

In the current workspace, `.venv` already exists and reuses the installed
PyTorch and MACE through system site packages, with its own e3nn and pytest
installation.
The shell's `PYTHONPATH` otherwise puts the user site packages ahead of this
environment. `env -u PYTHONPATH` affects only the invoked command. No existing
MACE packages were changed. The existing benchmark environment uses PyTorch
2.7.1, MACE 0.3.16, and e3nn 0.5.9. MACE 0.3.16 declares `e3nn==0.4.4` in its
[package requirements](https://github.com/ACEsuit/mace/blob/v0.3.16/setup.cfg).
The MAPHO requirement permits that version so a fresh installation can satisfy
both packages. Keep the recorded environment when reproducing existing
checkpoints; the dependency declaration does not migrate trained models.
For e3nn 0.4.4 with newer PyTorch, MAPHO permits the built-in `slice` type only
while importing the bundled Wigner constants. It leaves the serialization
defaults unchanged.
The CPU suite has been checked with both e3nn 0.4.4 and 0.5.9 using MACE
0.3.16 and PyTorch 2.7.1. GPU tests require a separate CUDA allocation.

The demo writes `model.pt`, `metrics.json`, `history.json`, `runtime.json`,
`split.json`, `heldout_sample.npz`, and `heldout_bands.npz`. Output directories
must be new. Run `python -m mapho --help` inside a suitable environment for
commands. An s+p+d example is:

```bash
env -u PYTHONPATH .venv/bin/python -m mapho demo \
  --basis spd --channels 4 --epochs 80 --output runs/spd_example
```

## Implemented

- Complete periodic cutoff graphs, including multiple images of the same atom,
  triclinic cells, partial periodicity, and onsite blocks.
- Species embeddings and O(3)-equivariant tensor-product message passing, with
  radial networks, smooth cosine cutoffs, and norm nonlinearities.
- Coupled tensor readout in complete s/p/d orbital shells. Presets `s`, `sp`,
  and `spd` contain 1, 4, and 9 orbitals per atom. The Python API also accepts
  repeated shells, such as `2x0e + 2x1o + 1x2e`.
- Exact reverse-edge Hermiticity: `H_ij(R) = H_ji(-R).T`. Onsite blocks are
  symmetric. The network predicts real spinless matrices; complex labels are
  rejected explicitly rather than silently cast.
- Structure-level train/validation/test splits, onsite/offsite balanced loss,
  validation-selected checkpoints, and held-out evaluation.
- Structure-only inference, standard Bloch diagonalization, and a Python API
  for generalized eigenproblems with a supplied positive-definite overlap.
- Strict NPZ training samples, flat-block HDF5 reading/writing, and a raw
  Wannier90 `*_hr.dat` reader that preserves complex entries and applies
  Wigner-Seitz degeneracy factors.

The implementation uses the tensor products documented in the
[e3nn API](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html).
No graph-learning framework is required beyond PyTorch and e3nn.

## Conventions and data contract

All atoms currently use the **same complete orbital basis**. Species-dependent
basis sizes, orbital masks, spin, SOC, and magnetic ordering are not implemented.
The basis uses e3nn's real spherical harmonics, including their ordering and
phases. A label source with different conventions needs an explicit basis
transformation; matching matrix dimensions alone is insufficient.

Rows of `cell` are lattice vectors in angstrom. Positions are Cartesian in
angstrom. Energies are eV. A block key is `[Rx, Ry, Rz, i, j]`, with integer
lattice translations and zero-based atom indices. The bond vector is
`positions[j] + R @ cell - positions[i]`. Every pair within the cutoff,
including its reverse and all onsite pairs, must have a label.

The Bloch convention is

```text
H_ij(k) = sum_R H_ij(R) exp(2 pi i k.R)
```

where k is in fractional reciprocal coordinates. This is the cell-periodic
gauge; atom-position phases are not included. Matrices transform under a
rotation or reflection Q as `H' = D(Q) H D(Q).T`.

`save_sample()` writes NPZ arrays `numbers`, `positions`, `cell`, `pbc`,
`pairs`, `hamiltonian`, and a JSON `metadata` string. The metadata records
schema version, units, basis, basis convention, cutoff, and spin status.
Training additionally requires the same nonempty `energy_reference`
description and the same boolean `orthogonal_basis` across the dataset.
These declarations record an upstream preprocessing contract; they do not
perform or prove gauge alignment.

```python
from ase import Atoms
from mapho.graph import make_graph
from mapho.data import Sample, save_sample

atoms = Atoms("CO", positions=[[0, 0, 0], [1.3, 0, 0]], cell=[8, 8, 8], pbc=True)
graph = make_graph(atoms, cutoff=4.0)
# labels: a real torch tensor (len(graph.pairs), 4, 4), ordered as graph.pairs.
# It must already be in the stated orbital basis and common energy gauge.
# sample = Sample(graph, labels, "sp", {
#     "energy_reference": "documented common reference",
#     "orthogonal_basis": True,
#     "source": "calculation identifier and export provenance",
# })
# save_sample("data/structure_001.npz", sample)
```

```bash
env -u PYTHONPATH .venv/bin/python -m mapho train \
  --data data --output runs/material --epochs 200
env -u PYTHONPATH .venv/bin/python -m mapho predict \
  --checkpoint runs/material/model.pt --structure structure.extxyz \
  --kpoints kpoints.txt --output predicted.npz
```

Use a structure format preserving PBC flags, such as extended XYZ. ASE reads
POSCAR as periodic in all three directions. Inference rejects elements absent
from training. There are no pretrained material parameters in this project.
For orthogonal labels, prediction also writes eigenvalues (Gamma by default).
For nonorthogonal labels, it writes Hamiltonian blocks only; use
`band_energies(H, graph, kpoints, overlap=S)` with independently supplied S.
An overlap model is not included.

## Raw adapters and plane-wave codes

`read_blocks_h5()` and `write_blocks_h5()` use `atom_pairs`,
`chunk_boundaries`, `chunk_shapes`, and `entries`. They transport matrices;
they do not infer orbital identities, units, atom order, or energy reference.

`read_wannier90_hr()` returns translations and dense complex matrices,
already divided by their Wigner-Seitz degeneracies. Its `bloch(kpoints)`
method implements the basic Fourier interpolation described by the
[Wannier90 file specification](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).
It does not include orbital-dependent `*_wsvec.dat` corrections; use the
appropriate Wannier90 interpolation when those corrections are required.

Existing routes to obtain Wannier90 input are documented by
[VASP](https://vasp.at/wiki/LWANNIER90),
[Quantum ESPRESSO](https://quantum-espresso.org/Doc/INPUT_pw2wannier90.html), and
[ABINIT](https://docs.abinit.org/tutorial/wannier90/). A raw `hr.dat` reader is
not an end-to-end interface to those codes. Independently optimized Wannier
functions can change phase, order, centre, and mix across structures. They
cannot automatically be treated as atom-centred irreducible orbital shells.
No automatic Wannier-to-training-sample conversion is provided.

## Development route toward plane-wave Hamiltonians

The first field export and frozen-potential solver are implemented in
`mapho.fields`, `mapho.qe.upf`, and `mapho.qe`. They are separate from the
orbital NPZ format above. See [the QE backend guide](docs/qe.md) for the data
contract, reproducible checks, and current restrictions.

The potential formulation of [Yuan et al., arXiv:2407.14379, Methods II.B–C](https://arxiv.org/pdf/2407.14379)
suggests a second model target: a smooth self-consistent potential residual,
decoded at spatial probes. The following describes the broader implementation
route. A first MACE/FNO field-learning pilot is now implemented; cross-code
integrations and the broader validations below remain future work.

1. Keep this orbital model as a controlled baseline. Introduce a separate
   atom-to-probe scalar decoder whose resolution is independent of the atomic
   graph and whose output can be sampled on each code's FFT grid. Chunk probe
   evaluation to control memory; split data by structure, never by grid point.
2. Define the target operator decomposition before exporting labels. Store
   potential components, ionic baseline, reference density, gauge, grid origin,
   units, functional, pseudopotential identity, charge, and spin metadata. Start
   with one nonmagnetic semilocal functional and norm-conserving pseudopotentials.
3. Reconstruct `H psi` using the exact kinetic operator, the local potential
   through FFTs, and the correct nonlocal projectors. Avoid a dense plane-wave
   Hamiltonian. Validate this reconstruction against the parent DFT code before
   replacing any component with machine learning.
4. Treat long-range electrostatics explicitly. Test charge redistribution,
   dipoles, interfaces, vacuum thickness, and screening; investigate a predicted
   density correction plus a periodic/slab Poisson solver. A finite local
   receptive field should not be assumed sufficient for these cases.
5. Add PAW/ultrasoft support as a distinct operator backend, with its overlap,
   augmentation, and state-dependent onsite terms. A local potential grid alone
   does not contain the full PAW information; see the
   [VASP formalism](https://vasp.at/wiki/Projector-augmented-wave_formalism).
   A basis-independent learned target is not automatically independent of
   pseudopotential or exchange-correlation convention.
6. Compare the orbital and potential approaches on identical DFT structures,
   gauges, and splits. Report bands near the Fermi level, gaps, density and
   subspace errors, SCF residuals, and wall time in addition to elementwise MAE.
   For MXenes, hold out termination, composition, strain, and defect families.
   Test a predicted potential as an SCF initializer before claiming a fully
   non-self-consistent replacement.

## Validation

Tests cover rotations, inversion, translation, atom permutation, batch
independence, multiple periodic images, triclinic cells, isolated atoms,
Hermiticity, analytic chain bands, complex generalized eigenproblems,
finite backpropagation, malformed data, adapter round trips, training, and
checkpoint reload. These checks establish implementation properties, not
transferability to materials outside the training distribution.

One CPU run (`runs/sp_demo`, seed 7, 40 synthetic structures, 80 epochs,
24/8/8 train/validation/test split, 50,850 parameters) achieved:

| Quantity | Result |
|---|---:|
| Initial validation Hamiltonian MAE | 38.89 meV |
| Best validation Hamiltonian MAE | 0.87 meV |
| Held-out Hamiltonian MAE | 0.79 meV |
| Held-out onsite / offsite MAE | 2.71 / 0.23 meV |
| Band MAE on one held-out structure, 51 k points | 14.59 meV |

These are analytic synthetic-task results. The band statistic concerns one
structure, not the whole test set. The example has a narrow structural range;
it is intended to verify train/predict/reconstruct, not establish accuracy on
DFT labels or support a state-of-the-art claim.


## Project name and stored data

MAPHO expands to **Many-body Potential-based Hamiltonian Operator**. Use
`import mapho`, `python -m mapho`, and the `mapho`, `mapho-qe`, `mapho-fields`,
and `mapho-vasp` commands. VASP commands are grouped into `data`, `train`,
`predict`, and `validate`; run `mapho-vasp --help` to start.

The old `mlham` Python namespace and import aliases have been removed.
Readers still accept stored data with the former schema prefix, and QE
restart lookup reads the prefix from the saved input. New VASP calculations use
`MAPHO_EXPORT`, `MAPHO_IMPORT`, and `MAPHO_PAW.dat` with the matching private
executable. Old studies require their saved executable and source snapshot.
Historical runs, source snapshots, and hashes are preserved. The `runs`
directory links to scratch. See the [package layout](docs/backends.md) and
[migration notes](docs/renaming.md).
