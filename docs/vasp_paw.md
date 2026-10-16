# VASP PAW: scalar and collinear PBE operator interface

An initial [PBE0 component pilot](vasp_pbe0.md) now tests the additional
exchange requirement. Its partial hybrid exports are explicitly separated
from the complete PBE operators described here.

This backend exports a converged VASP PAW operator and supplies it to VASP's
native generalized eigensolver. The nonmagnetic acceptance study uses Si and
LiF; the collinear `ISPIN=2` study uses ferromagnetic bcc Fe.
Those acceptance studies validate the interface, not a converged materials
benchmark. A separate joint Si16 training pilot is described below. The
existing QE checkpoints are not PAW models.

The smooth potential is only one part of the PAW Hamiltonian:

\[
\widetilde H=T+\widetilde V+\sum_{aij}|p_i^a\rangle D_{ij}^a\langle p_j^a|,
\qquad
S=1+\sum_{aij}|p_i^a\rangle Q_{ij}^a\langle p_j^a|.
\]

VASP retains its kinetic energy, PAW projectors, overlap, and plane-wave solver.
The interface supplies the total smooth potential and full on-site D matrices.
It verifies the exported Q matrices against those generated natively from the
same PAW datasets; it never replaces Q. The basis conventions are native VASP
projector order. The Si training head now maps complete s/p shells to e3nn;
other projector angular momenta are not yet supported by that head.

## Native bridge

The local VASP 6.6.1 GPU build has no HDF5 support. A small opt-in bridge is
compiled in an isolated scratch copy; the installed executable stays unchanged.
Only mapho's new bridge and an anchor-checked patching script are distributed
here. VASP source, executables, and POTCAR files remain private local inputs.

`python -m mapho.vasp.vasp_patch PRIVATE_BUILD_DIRECTORY` modifies `pot.F` and
`main.F`, adds `vasp_bridge.F.inc`, and records before/after SHA256 hashes.
It refuses already-patched or incompatible source anchors. This bridge has
only been targeted at 6.6.1; its anchors are not a compatibility promise for
other versions. Recompile `pot.o`, `main.o`, and the dependent objects/link
using that build's original compiler and MPI configuration.

- `MLHAM_EXPORT` in a calculation directory enables native `POT` output and
  full-precision `MLHAM_PAW.dat` D/Q output.
- `MLHAM_IMPORT` with `ICHARG=14` enables full D import after native `POT`
  reading. The bridge checks dimensions, atom types, radial angular channels,
  symmetry, finite values, and native Q consistency.
- Native `POT` contains an augmentation expansion as well as the grid. The
  bridge overrides that expansion's reconstructed D with the full matrix,
  avoiding angular truncation/serialization error in the expansion.
- The pilot supports nonmagnetic (`ISPIN=1`) and collinear (`ISPIN=2`) PBE,
  with `KPAR=1`, `LREAL=.FALSE.`, and
  `LH5=.FALSE.`. The generated inputs also use `NCORE=1` and `ISYM=-1`.
  SOC, noncollinear magnetism, hybrids, meta-GGAs, DFT+U, forces, and stress are
  outside this interface's validated scope. Only the single-GPU layout is
  included in the initial acceptance study.
  The reported acceptance metrics concern eigenvalues. A supplied potential
  and D do not define an ML total-energy functional; frozen-run total energies
  are not used as learned energy predictions.

VASP documents potential input through
[`ICHARG=4`](https://vasp.at/wiki/index.php/ICHARG), and adding 10 freezes the
charge during electronic minimization. The 6.6.1 source path for `ICHARG=14`
also preserves the supplied potential and on-site operator. The native replay
test checks this behavior rather than assuming an ordinary frozen-charge run
is equivalent to supplying a learned potential.

## Python interface and data

`mapho.vasp.paw.prepare_reference` accepts ASE atoms, per-element local PAW
dataset paths, ENCUT, an explicit Gamma-centered k mesh, and NBANDS. Inputs,
dataset titles, hashes, and electron counts are recorded in `manifest.json`.
It rejects non-PBE/non-PAW datasets and noncollinear spin vectors. The numerical
settings are pilot defaults and require convergence checks before training.
`algorithm='Exact'` is available for small operator acceptance tests; the
default remains the iterative `Normal` solver.

`run` executes the selected native executable, hashes it, checks the inputs,
and requires VASP completion, electronic convergence, valid bands, and the
expected bridge marker. Use it inside a scheduler allocation.

```python
from mapho.vasp.paw import export_operator, prepare_replay, run, compare

export_operator("reference", "operator.h5")
prepare_replay("operator.h5", "replay")
run("replay", "/private/path/to/patched/vasp")
print(compare("reference", "replay"))
```

The HDF5 schema `mapho.vasp_paw_operator.v1` stores (the historical
`mlham.vasp_paw_operator.v1` identifier remains readable):

- `potential_eV`: the total smooth potential, shape `(nx,ny,nz)`, sampled at
  fractional coordinates `(ix/nx,iy/ny,iz/nz)`. Native POT has x varying
  fastest. No CHGCAR volume normalization is applied.
- `paw/d`, `paw/q`: full matrices in native VASP normalization, with shapes
  `(natoms,lmdim,lmdim)`. D has energy units; Q enters the dimensionless
  overlap. Inactive channels are zero-padded.
- `paw/types` and `paw` attribute `channels_json`: zero-based species indices
  and the ordered radial-projector angular momenta for each species.
- Cell, atomic positions/numbers, reference eigenvalues/occupations/k points,
  native POT header and augmentation suffix, and source/executable hashes.

The geometry stored in HDF5 and used for replay is the full-precision hashed
POSCAR. Native POT headers round positions/cell entries and may wrap atoms to
equivalent periodic images; the exporter checks those conventions explicitly.
Potential files use VASP's fixed-width grid format, which is stricter than a
generic whitespace-separated volumetric file.

`prepare_replay(..., potential=predicted_grid, d=predicted_D)` is the injection
API for subsequent learning. A replay can use new fractional reciprocal
`kpoints`; it needs no reference WAVECAR or CHGCAR. The initial API uses the
reference directory for the matching POSCAR/POTCAR and checks their hashes.

## Acceptance study

The scratch study is `runs/vasp_paw_pbe_v5`, with a source snapshot, validation
SLURM script, native input/output files, operator HDF5 files, and
`validation.json`. The private build and its hashes are in
`runs/vasp_paw_pbe_v1/build/std`. Attempts v1/v2 retain their MPI startup failure
logs; they produced no DFT results. The installed NVHPC OpenMPI runtime needs
`OPAL_PREFIX` pointing to its installed MPI directory and an explicit
`mpirun -np 1` launcher. The validation script records the working environment.
Attempt v3 exposed a fixed-width POT serialization error. Attempt v4 established
the correct PAW gauge response and new-k replay, but the highest empty bands
of the iterative SCF reference were underconverged and LiF needed periodic
image comparison in its structure check. Those records are retained.
The two-atom cells use 600 eV, a 2x2x2 SCF mesh, PBE54 Si,
Li_sv, and F datasets, and 12/16 bands. Full diagonalization (`ALGO=Exact`)
is used in this acceptance study to avoid incomplete empty-band convergence.
The six-decimal `EIGENVAL` energies are cross-checked against the four-decimal
XML values; errors below 1 micro-eV are not resolved by these text outputs.
These small calculations test the
interface; their meshes are not claimed converged.

The acceptance threshold is 0.1 meV maximum absolute eigenvalue error over
all requested bands/k points. Checks cover:

1. Unmodified VASP SCF versus the bridge-enabled SCF.
2. Exported operator replay from random orbitals at the SCF k points.
3. A +0.25 eV gauge shift applied to both V and D: `V += c`, `D += c Q`.
   Every eigenvalue should move by c because `delta H = c S`.
4. A negative control that shifts V alone, omitting the PAW contribution.
5. Three k points absent from the SCF mesh: operator replay versus unmodified
   VASP's independent `ICHARG=11` calculation using the converged CHGCAR.

The automated driver is `mapho-vasp predict paw_validation STUDY_ROOT`.
It refuses to overwrite existing runs. `validation.json` is the authoritative
record of which native checks completed and passed.

Completed on 2026-10-15, debug-gpu job **16255950**, one A100, VASP 6.6.1.
All fourteen native calculations completed; both materials passed every
acceptance check. Maximum absolute eigenvalue errors in meV:

| Check | Si (12 bands) | LiF (16 bands) |
| --- | ---: | ---: |
| Bridge SCF versus unmodified VASP | 0.000 | 0.000 |
| Operator replay at 8 SCF k points | 0.001 | 0.002 |
| Consistent +0.25 eV gauge shift | 0.001 | 0.002 |
| Replay at 3 new k points versus native frozen-charge solve | 0.001 | 0.001 |
| Negative control: smooth potential shifted without D | 55.906 | 240.044 |

The negative control is expected to fail the uniform-spectrum-shift test;
its failure is required for overall acceptance. These numbers validate
serialization, units, PAW D/Q handling, and native operator replay. They do
not measure a learned model's predictive accuracy. Ten Python regression
tests also pass (`tests/test_vasp_paw.py`).

Reference/replay directories contain `INCAR`, `POSCAR`, `POTCAR`, `KPOINTS`,
manifests, executable hashes, stdout, `OUTCAR`, `EIGENVAL`, and `vasprun.xml`.
Each material has an `operator.h5`. Replay directories contain no reference
`CHGCAR` or `WAVECAR`; only the independent native frozen-charge controls
receive `CHGCAR`. VASP creates an empty `WAVECAR` placeholder in replay
directories; stdout explicitly confirms that wavefunctions were not read.

## Collinear spin extension

For scalar-relativistic collinear PBE without SOC, the two spin channels solve
separate generalized eigenproblems with the same overlap:

\[
H_\sigma=T+V_\sigma+\sum_{aij}|p_i^a\rangle D^a_{ij,\sigma}\langle p_j^a|,
\qquad H_\sigma\psi_{n\sigma}=\epsilon_{n\sigma}S\psi_{n\sigma}.
\]

This matches VASP's separate up/down treatment for
[`ISPIN=2`](https://vasp.at/wiki/index.php/ISPIN). The native POT writer stores
up/down potentials, not the charge/magnetization arrangement used in CHGCAR.
Both fields and both on-site matrices must be preserved. The bridge checks
that VASP's Q matrices agree between spins and stores a single shared Q.

```python
from ase.build import bulk
from mapho.vasp.paw import prepare_reference, prepare_replay

prepare_reference(
    "Fe_reference", bulk("Fe", "bcc", a=2.86),
    {"Fe": "/private/potentials/Fe/POTCAR"},
    encut=600, mesh=(2, 2, 2), nbands=12,
    ispin=2, magmoms=[2.5], algorithm="Exact",
)
# After running the reference and exporting operator.h5:
prepare_replay("operator.h5", "replay")
prepare_replay("operator.h5", "spin_reversed", reverse_spins=True)
prepare_replay("operator.h5", "spin_shift", shift_eV=[0.17, -0.09])
```

`magmoms` supplies one signed initial moment per atom. Alternatively, the
preparer reads ASE's initial magnetic moments. Optional `nupdown` fixes the
up-minus-down electron count using VASP's
[`NUPDOWN`](https://vasp.at/wiki/index.php/NUPDOWN); omitting it allows the
occupations to relax. The pilot conservatively requires enough bands for a
fully polarized state when that constraint is omitted. `MAGMOM` initializes
the SCF state; it does not guarantee its converged magnetic order.

The spin HDF5 format is `mapho.vasp_paw_operator.v2` (the historical
`mlham.vasp_paw_operator.v2` identifier remains readable):

| Quantity | Shape / convention |
| --- | --- |
| `potential_eV` | `(2,nx,ny,nz)`, index 0 up and index 1 down |
| `paw/d` | `(2,natoms,lmdim,lmdim)`, same spin order |
| `paw/q` | `(natoms,lmdim,lmdim)`, shared native overlap correction |
| `bands/eigenvalues_eV`, `bands/occupations` | `(2,nkpoints,nbands)` |
| `bands/spin_electrons` | Two k-weighted occupation sums |
| `bands/spin_moment_muB` | Difference of those occupation sums |
| `native_pot_suffixes` | Two native augmentation/separator records |

The matrix text format is `MLHAM_PAW_V2`; each active matrix element contains
`D_up D_down Q`. V1 scalar files remain supported with their original shapes.
Spin reversal exchanges both V and D and reverses the initial moments; it
refuses nonzero fixed `NUPDOWN` constraints rather than silently changing one.
Shifts are applied as `V_sigma += c_sigma`, `D_sigma += c_sigma Q`. Equal
shifts are a common gauge change; unequal shifts change the spin splitting.

The collinear acceptance driver is `mapho-vasp validate spin ROOT`.
The scratch study `runs/vasp_paw_spin_v1` contains the private extended build,
input files, source snapshot, full outputs, and `validation.json`. It uses
one-atom bcc Fe at 2.86 A, 600 eV, 2x2x2 k sampling, 12 bands per spin,
Gaussian smearing of 0.05 eV, and exact diagonalization. The mesh is deliberately
small: neither its moment nor its band energies are claimed converged material
properties, and these data are not used for model training.

Checks cover an unmodified spin-polarized SCF reference; replay of both spins;
common and unequal spin shifts; operator spin reversal; an independent native
SCF initialized with opposite moments; three new k points compared with native
frozen-charge calculations; and replay of the existing scalar Si V1 operator.
For total spin moments the report uses k-weighted occupations. A frozen-POT
calculation does not update its starting density, so its printed density-derived
magnetization is not used as the moment of the solved eigenstates.

Completed on 2026-10-15, debug-gpu job **16256096**, one A100, VASP 6.6.1.
All ten native calculations completed in a 3 min 33 s allocation; every
acceptance check passed. Maximum absolute eigenvalue errors over both spins:

| Check | Error (meV) |
| --- | ---: |
| Bridge SCF versus unmodified VASP | 0.000 |
| Fe operator replay at 8 SCF k points | 0.001 |
| Common +0.25 eV shift | 0.001 |
| Independent +0.17 / -0.09 eV spin shifts | 0.001 |
| Reversed operator versus exchanged reference channels | 0.001 |
| Independent native SCF with reversed initial moments | 0.000 |
| Replay at 3 new k points versus native frozen-charge solve | 0.004 |
| Existing Si V1 operator replay with the extended bridge | 0.001 |

The occupation-derived moment also reverses within 1.3e-7 Bohr magnetons.
All eighteen Python tests pass across `tests/test_vasp_paw.py` and
`tests/test_vasp_spin.py`. As in the scalar study, replay starts without
reference wavefunctions or charge density. These results validate operator
transport and spin conventions, not magnetic ML prediction accuracy.

## Learning work after interface validation

The Si16 pilot reuses the existing MACE/FNO machinery and adds an equivariant
on-site head for residual D, including repeated radial channels. Its s/p
harmonic mapping and geometry-only atomic reference are checked below.
Targets retain the native VASP gauge: the model learns both absolute
residuals, including their scalar offsets. Exporting/replaying a reference
alone does not establish learned predictive accuracy.

For magnetic learning, geometry alone does not identify every possible
magnetic state. Training must specify the intended branch (for example FM or
AFM) or supply suitable spin-state inputs. A useful output decomposition is
`V0=(V_up+V_down)/2` and `B=(V_up-V_down)/2`, and the analogous decomposition
of D. Under reversal of all moments, V0/D0 remain unchanged and B/D_spin
change sign. The current work validates transport/replay of spin-resolved
operators; a magnetic MACE/FNO model and AFM validation remain future work.

## Si16 reference-data pilot

`mapho.experiments.vasp.silicon_dataset` prepares 24 independent nonmagnetic PBE Si
configurations. Each is a 2x2x2 replica of the two-atom diamond primitive
cell, giving 16 atoms. Starting from a=5.43 A, configurations receive
isotropic strain within +/-2%, symmetric strain components within +/-0.5%,
and independent atomic displacement components within +/-0.04 A before
removing the collective translation. The geometry seed is 29; the fixed
structure-level split is 16 training, 4 validation, and 4 test configurations.

Pymatgen writes POSCAR, INCAR, and KPOINTS. All production configurations use
`Kpoints.automatic_density_by_vol(structure, 500, force_gamma=True)`, which
gives 7x7x7 meshes for this pilot. `kppvol` is the requested number of mesh
points per reciprocal-cell volume in inverse cubic angstroms; it is not
the number of points per atom or a reciprocal spacing. Integer rounding
means the realized density differs from the requested value. Both values,
the generated mesh, and the pymatgen version are recorded per configuration.
See the [pymatgen API](https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.inputs.Kpoints.automatic_density_by_vol).

The scratch study is `runs/vasp_si16_v1`. Its production inputs are in
`samples/sample_000` through `samples/sample_023`; the calibration inputs are
under `convergence/`. All use PBE54 Si, ENCUT=600 eV, 48 bands, ALGO=Normal,
EDIFF=1e-9 eV, Gaussian smearing of 0.05 eV, and at least 12 SCF steps.
The native PAW bridge settings remain explicit. WAVECAR output is disabled;
CHGCAR is retained, along with POT and the full on-site matrices.

Before production, a separate pristine geometry and a compressed/distorted
geometry are evaluated at kppvol=200, 300, and 500 (5x5x5, 6x6x6, 7x7x7).
For both successive mesh refinements and both geometries, the gate requires:

- Energy changes below 1 meV/atom.
- Four valence and four conduction bands around the gap at common k points
  to change by at most 1 meV, with Gamma required.
- Smooth-potential and active PAW D-element RMS changes below 5 meV.

No fitted gauge shifts are used. Sampled gap changes are recorded separately:
different SCF meshes sample different extrema, so they are not used to claim
convergence of the fundamental band gap. These checks establish sampling
accuracy at a fixed 600 eV cutoff; they do not establish cutoff convergence.
Calibration structures are excluded from all three dataset splits.

`references.sbatch` uses the existing validated GPU executable. One debug-gpu
allocation supplies four GPUs, with one independent VASP process assigned to
each visible device. The driver runs calibration first and launches production
only if `convergence.json` passes. Worker failures stop the workflow from
declaring completion. `completion.json` is written only after all 24 exported
`operator.h5` files pass the worker checks. Native inputs, outputs, hashes,
source snapshots, and individual logs remain in scratch.

Job **16256500** completed successfully on 2026-10-15 in **51 min 24 s**
using four A100 GPUs. All six calibration calculations and all 24 production
references completed and exported their operators. Both 5x5x5 to 6x6x6 and
6x6x6 to 7x7x7 comparisons passed on both calibration geometries. The largest
changes for the latter refinement were:

| Quantity | Largest change across the two geometries |
| --- | ---: |
| Energy | 0.05925 meV/atom |
| Frontier levels at Gamma (the common k point) | 0.081 meV |
| Smooth-potential RMS | 0.07535 meV |
| Active PAW D-element RMS | 0.00358 meV |

`data_audit.json` verifies all 30 run statuses, input and export-source hashes,
operator hashes, finite values, D/Q symmetry, and split membership. The 24
production HDF5 files occupy about 382 MB combined. Eighteen configurations
have 128x128x128 grids; six have some or all dimensions equal to 120. A training
loader must retain each configuration's cell and native grid dimensions.
The on-site arrays are `(16,8,8)`, and each reference stores 48 bands at 343
k points. Sampled gaps span 0.3073 to 0.5713 eV; these are mesh-sampled values,
not a claim of converged fundamental gaps. This data-generation study does
not itself train a VASP PAW model.

This reference-data pilot supplies the training study below. The small
supercells do not establish performance on long-range transfer tasks.

## Joint Si16 PAW training pilot

The implementation is in `vasp_model.py`, `vasp_train.py`, and
`vasp_training_validation.py`. Atomic references are generated by
`vasp_atomic_reference.py`, using VASP's
[`ICHARG=12`](https://vasp.at/wiki/index.php/ICHARG): the superposed atomic
densities remain fixed during a short non-SCF calculation. Every reference
starts with ISTART=0 and no input CHGCAR or WAVECAR. Only geometry, matching
PAW datasets, settings, and the numerical grid are needed. These baseline
calculations are stored in `runs/vasp_si16_atomic_v1` (24 completed, job
16256702); they are not converged PBE inputs.

One shared, trainable MACE backbone feeds a continuous atom-centered
potential decoder and a symmetric PAW matrix head. For Si's s/s/p/p radial
channels, the latter spans all 36 independent components of an 8x8 real
symmetric matrix. Linear and quadratic equivariant readouts include scalar,
polar-vector, axial-vector, and rank-two channels. Native p_y,p_z,p_x order
is mapped to e3nn p_x,p_y,p_z without mixing radial channels. Tests check
matrix symmetry, completeness, rotations, atom permutations, and gradients.
Native VASP checks on a distorted training geometry reproduce rotated D
within 7.2e-15 eV and rotated potentials within 1.1e-9 eV; an independent
k-point change gives the same atomic-reference operator to those tolerances.

The field loss samples 8192 points from each native FFT grid. This avoids
replacing the potential target with a coarse-grid low-pass field. The local
decoder is evaluated continuously at those points; the FNO correction uses
a 32x32x32 grid and periodic trilinear interpolation, identically in training
and inference. The three FNO input channels are the local predicted residual,
the total atomic-reference potential, and a geometric Gaussian atom envelope.
No converged electronic field, target mean, target D, or eigenvalue is supplied
as an inference input. Full-grid test errors use every native grid point.

Normalization uses the 16 training structures only. The measured mean MACE
neighbor count is 20.703125 rather than the earlier fixed value 16. The loss
balances field and on-site MSE using their training-set residual RMS scales
(0.38956 eV and 0.013292 eV respectively). The backbone, potential readout,
matrix readout, and FNO all receive optimizer updates. Four validation
structures select the checkpoint; the four test structures are reserved for
evaluation. No per-test gauge alignment is used.

`runs/vasp_si16_train_v1/train.sbatch` ran job 16256719 on three A100s.
It fitted MACE alone (`local`), MACE with the FNO's spectral branch disabled
(`none`), and full MACE/FNO (`full`), with shared seed 31 and identical common
initialization. The latter two have identical pointwise depth and width.
Each fit allows up to 300 epochs, validates every five epochs, reduces the
learning rate on validation plateaus, and stops after 100 epochs without a
new best validation score once at least 150 epochs have run. A single seed
is a pilot, not an uncertainty estimate or an architectural ranking.

After fitting, the job reconstructs both V and D on each test geometry and
runs VASP ICHARG=14 at Gamma and two off-mesh k points. Atomic-reference,
MACE, pointwise, and FNO operators are compared against an independent native
ICHARG=11 calculation. Reference-operator replay is checked separately to
bound serialization/solver errors. The report covers eight frontier bands
and all 48 calculated bands. These calculations do not determine a converged
fundamental gap. Checkpoints, histories, field metrics, predictions, native
replays, source snapshots, and final `completion.json` remain in scratch.

Job **16256719 completed successfully in 9 min 40 s** on 2026-10-15,
including training, full-grid prediction, and native band validation. Each
model completed 300 epochs (4800 optimizer updates). Results on the four
held-out structures are:

| Model | Potential RMSE (eV) | PAW D-element RMSE (meV) | Frontier-band MAE (meV) | All-48-band MAE (meV) |
| --- | ---: | ---: | ---: | ---: |
| Fixed atomic reference | 0.39374 | 13.769 | 388.08 | 422.15 |
| MACE | 0.10914 | 0.758 | 19.58 | 14.30 |
| MACE + pointwise (`none`) | 0.07081 | 0.754 | 2.76 | 2.87 |
| MACE + FNO (`full`) | 0.05919 | 0.748 | 1.66 | 1.77 |

Field and D errors are the square root of the mean per-structure MSE.
Frontier errors average four valence and four conduction levels at each of
three k points on each test structure (96 levels); no fitted energy shifts
are applied. Native reference-operator replay has a frontier MAE of
0.00233 meV, much smaller than the ML errors. Both V and D are predicted
in the ML band solves; only the independent native reference uses the
converged CHGCAR. Inference needs a fixed atomic-density calculation and
the subsequent band solve, but no converged electronic input.

All three validation minima occurred at the final epoch, so these are
budget-limited pilot results, not converged architecture comparisons. The
pointwise and FNO corrections have the same width (8) and depth (2), and
identical initial common parameters; the FNO additionally trains its spectral
filters. Total trainable parameter counts are 13,667, 13,852, and 16,092 for
MACE, pointwise, and FNO respectively. Training peak GPU allocation is about
5.6 GiB per fit, including cached decoder bases. Multiple seeds, longer fits,
and broader structures remain necessary before claiming a reliable FNO
advantage or transfer beyond these mildly distorted Si16 cells.

The final artifact audit is `runs/vasp_si16_train_v1/training_audit.json`;
the FNO checkpoint is `runs/vasp_si16_train_v1/full/model.pt`. The audit
checks source and input hashes, all 26 native validation runs, prediction
finiteness and matrix symmetry, checkpoint contracts, shared initialization,
and actual updates to the MACE, on-site, field, and spectral parameters.

### Transfer to primitive silicon

`vasp_primitive_bands.py` applies the three unchanged Si16 checkpoints to
ideal two-atom diamond Si at a = 5.43 Angstrom. Study
`runs/vasp_si2_bands_v1` (job **16256772**, three A100s, **10 min 43 s**)
completed on 2026-10-15. It creates an independent fixed-atomic-density
reference, predicts both V and D without electronic labels, and evaluates
eight bands at 81 points along X-Gamma-L. X and L have primitive reciprocal
coordinates (0.5,0,0.5) and (0.5,0.5,0.5), respectively. There is no
retraining, changed normalization, or fitted energy alignment.

Pymatgen reciprocal-volume densities of 500 and 750 give 14x14x14 and
16x16x16 Gamma-centered SCF meshes. Refinement changes the energy by
0.01301 meV/atom, the potential RMS by 0.03244 meV, and the largest frontier
level by 0.019 meV. The denser mesh supplies the native ICHARG=11 reference.
Exact-operator replay agrees with it to 0.001 meV in the frontier window.
All calculations retain the training cutoff of 600 eV.

| Model | Potential RMSE (eV) | Frontier-band MAE (meV) | Path-sampled gap (eV) |
| --- | ---: | ---: | ---: |
| Native PBE | — | — | 0.572421 |
| MACE | 0.10664 | 21.785 | 0.624867 |
| MACE + pointwise | 0.06797 | 1.053 | 0.572669 |
| MACE + FNO | 0.05487 | 2.369 | 0.566124 |

Here the frontier window contains bands 3-6 (two valence and two conduction)
at all 81 points, rather than the eight-band window of the Si16 test. The
VBM is at Gamma and the sampled CBM is at 0.85 of Gamma-X for native PBE
and all three learned models. The pointwise model has smaller band errors
on this crystal despite the FNO's lower potential error. This single
primitive-cell transfer test is not an architecture ranking or a general
Brillouin-zone gap search.

`vasp_primitive_plot.py` produces `primitive_si_bands.pdf`, a PNG preview,
`band_source_data.csv`, and a caption. Both plotted dispersions use the
same native PBE VBM zero; the second panel displays signed frontier-level
errors. `results.json`, `validation_audit.json`, native inputs/outputs,
predicted fields, frozen-checkpoint hashes, and source snapshots remain
with the run. Rendered text and collision checks passed.

### Correction-activation control

`vasp_activation_ablation.py` prepares a paired SiLU/GELU comparison using
the existing training cache and native Si16/primitive references. The GELU
setting changes only the activation after each of the two correction-layer
sums: after the pointwise linear transform in `none`, and after the sum of
pointwise and spectral terms in `full`. GELU uses PyTorch's exact
`approximate='none'` implementation. MACE and the radial-filter MLPs retain
SiLU. The default remains SiLU for existing models.

`correction_activation` is stored in the training manifest and model
description. Loading checks these agree; older checkpoints without this
field retain SiLU. The two GELU fits start from the same seed-31 initial
weights as their SiLU controls, rather than modifying trained SiLU weights.
Data splits, cached grid points, input scaling, architecture, optimizer,
scheduler, loss, and 300-epoch budget are unchanged. MACE, both PAW readouts,
and the correction network remain jointly trainable.

Study `runs/vasp_si16_gelu_v1` uses job **16256821** on two A100s. Its
`compatibility_audit.json` records identical original/current SiLU outputs
and initial weight hashes. The targeted field, correction, and PAW model
tests passed (23 tests), including exact GELU evaluation, unchanged spectral
filter activations, gradients, spatial independence of the pointwise branch,
and checkpoint activation round trips. Native evaluations reuse the same
reference eigenvalues, without energy alignment or new SCF calculations.

Job **16256821 completed successfully in 7 min 57 s**. Both GELU models
completed 4800 updates and selected epoch 300, with validation still improving.
All ten new native band calculations completed. The fixed-budget results are:

| Model | Activation | Si16 potential RMSE (eV) | Si16 frontier MAE (meV) | Primitive frontier MAE (meV) |
| --- | --- | ---: | ---: | ---: |
| MACE + pointwise | SiLU | 0.07081 | 2.759 | 1.053 |
| MACE + pointwise | GELU | 0.06804 | 2.273 | 1.122 |
| MACE + FNO | SiLU | 0.05919 | 1.660 | 2.369 |
| MACE + FNO | GELU | 0.06151 | 2.103 | 0.953 |

The Si16 metric averages eight frontier levels at three k points on each
of four held-out structures. The primitive metric averages four frontier
levels at 81 X-Gamma-L points. Compare activations within each column;
these two columns have different evaluation windows. Primitive sampled gaps
are 0.569832 eV (pointwise/GELU) and 0.570066 eV (FNO/GELU), versus the
same native PBE reference of 0.572421 eV. PAW D-element RMS errors on Si16
are 0.7430 and 0.7410 meV for pointwise/GELU and FNO/GELU, respectively.

GELU helps the pointwise model on Si16 but helps FNO on the primitive cell;
the potential-error ordering does not consistently predict the band-error
ordering. This single paired seed does not justify a general activation
ranking. The SiLU default is retained, and GELU is explicitly selectable.
`completion.json`, `comparison.csv`, `validation_history.csv`, and
`validation_audit.json` preserve the results and checks. Source snapshots,
initial-state fingerprints, both GELU checkpoints, predictions, and native
VASP input/output files are retained in the study directory.

### Binary bulk and bilayer pilots

The generic PAW readout now supports multiple species with complete s/p/d
projector shells. It reads the radial channels from the exported operator,
creates an element-specific equivariant symmetric-matrix head, and masks
inactive native padding in both the loss and reported on-site errors. The
native d convention requires an orthogonal basis transformation, including
linear combinations, rather than only the p-shell permutation. Legacy Si
checkpoints retain their original readout. This extension passed 47 targeted
tests, including polynomial harmonic conventions, proper rotations, inversion,
atom permutations, parameter gradients, padding masks, rectangular grids,
checkpoint loading, and geometry-only prediction. Native convention/replay
checks for the new materials remain mandatory gates in their training jobs.

`vasp_material_dataset.py` prepares two separate datasets, each with 24
configurations and the same seed-29 16/4/4 structure split as the Si pilot:

| Dataset | Geometry | PAW datasets | Calibration meshes (pristine) |
| --- | --- | --- | --- |
| `vasp_lif16_v1` | Rocksalt LiF, 2x2x2 primitive, 16 atoms, a=4.03 A | Li_sv / F | 7x7x7, 8x8x8, 9x9x9 |
| `vasp_mos2_bilayer24_v1` | 2H bilayer, 2x2 in-plane, 24 atoms, a=3.18 A | S / Mo_pv | 8x8x1, 9x9x1, 10x10x1 |

These are ideal reference geometries with small perturbations, not relaxed
PBE equilibrium structures. Both use nonmagnetic PBE, 600 eV, no SOC or added
dispersion, EDIFF=1e-9, ISYM=-1 and LREAL=False. LiF uses 56 bands; MoS2 uses
112. Pymatgen reciprocal-volume densities are 200/300/500 for LiF and
500/750/1000 for MoS2. Each material uses one common final density across its
production configurations. The slab normal is sampled only at Gamma. Sampling
must pass energy, common-k frontier, potential-field and active-PAW-matrix
checks on pristine and distorted calibration structures before production.
Cutoff convergence is a separate question.

The bilayer has 3.19 A monolayer thickness, 6.15 A ideal Mo-plane separation,
and 20 A nominal outer-surface vacuum. Its in-plane strain and atomic
perturbations follow the Si amplitudes; layer separation additionally varies
by up to 2%. The cell height is fixed, so the perturbations do not strain the
vacuum. The Hamiltonian correction still uses the existing periodic 3D FNO,
which already mixes different heights. A 32x32x64 grid maintains finer normal
resolution for both pointwise and FNO corrections. This is not a new explicit
height-mixing slab architecture. Vacuum/grid convergence and primitive-cell
transfer checks are not yet completed for these materials.

`vasp_material_training.py` chains geometry-only ICHARG=12 atomic references,
training-only normalization, native rotation/k-independence checks, five fits,
and held-out VASP replays. The fits are MACE alone, pointwise/SiLU, FNO/SiLU,
pointwise/GELU and FNO/GELU. All retain the seed-31, 300-epoch budget, shared
MACE training, width 8, depth 2, fixed 8192 sampled native loss points, and
validation-based checkpoint selection. The two activations have identical
initial parameters. Eight frontier bands are evaluated at three common
k points on four held-out structures, using an independent native frozen-density
solve and an exact exported-operator replay. Slab evaluation points have kz=0.

All inputs, frozen sources and outputs are in scratch via `runs/`. Reference
jobs use four independent GPUs per allocation. Production is split into two
12-configuration allocations using `vasp_reference_chunk.py` to fit the debug
time limit. Current dependency chains are:

| Material | Convergence | Production chunks | Atomic references, training and bands |
| --- | --- | --- | --- |
| LiF | 16257022 | 16257031, 16257032 | 16257035 |
| MoS2 | 16257024 | 16257033, 16257034 | 16257036 |

The training directories are `vasp_lif16_train_v1` and
`vasp_mos2_bilayer24_train_v1`. Failed convergence prevents production;
failed data/native checks prevent fitting. Submitted jobs do not constitute
completed validation or evidence that either correction is more accurate.

### Binary pilot status on 2026-10-16

All 48 production reference calculations and both six-case k-point calibrations
completed successfully. Both data audits passed. All 48 atomic references and
all ten 300-epoch fits also completed. Native s/p/d rotation and k-independence
checks passed for both materials.

The LiF training/band job 16257035 completed successfully in 13 min 26 s.
Its four-structure native replay check has a maximum frontier error of
0.014 meV. The checkpoint/input/data hash audit in `validation_audit.json`
passed; results are in `comparison.csv` and `completion.json`:

| LiF model | Potential RMSE (eV) | PAW D RMSE (meV) | Frontier band MAE (meV) |
| --- | ---: | ---: | ---: |
| MACE alone | 0.49561 | 6.934 | 28.670 |
| Pointwise / SiLU | 0.40476 | 9.171 | 35.368 |
| FNO / SiLU | 0.36948 | 8.130 | 12.570 |
| Pointwise / GELU | 0.40455 | 8.938 | 28.852 |
| FNO / GELU | 0.36362 | 8.569 | 15.535 |

These are eight frontier bands at three k points for four held-out structures,
with one paired seed and no fitted energy shift. FNO lowers the mean band error
at this budget for both activations; GELU does not consistently improve the
band error. These data do not establish long-range superiority.

The MoS2 job 16257036 exited with a failed band-validation gate after all fits
and all 32 band calculations completed. On `sample_023`, exact exported-operator
replay differs from the native frozen-density solve by up to 1.048 meV, exceeding
the fixed 1 meV frontier threshold. The other three replay checks passed.
In addition, predicted operators on `sample_020` show spurious low-energy states;
some local/pointwise predictions on `sample_023` also fail spectrally. Small
field or matrix RMSE therefore does not establish useful Hamiltonian accuracy.
Its `validation_audit.json` is explicitly failed and its aggregate metrics are
labelled `comparison_unvalidated.csv`; there is no successful completion claim.

Diagnostic job 16258661 in `runs/vasp_mos2_operator_diagnostic_v1` separates
learned-V/exact-D and exact-V/learned-D errors, repeats solves with EDIFF=1e-11
and NELMIN=24, and exports/replays the native frozen-density Hamiltonian to
separate reconstruction changes from bridge accuracy. It uses the existing
checkpoints without retraining, changing test structures, or relaxing thresholds.

The diagnostic confirmed two separate issues. Freshly exporting the native
ICHARG=11 operators and replaying them reproduces all printed eigenvalues for
both inspected structures (zero difference at EIGENVAL precision). Reconstructing
an operator from CHGCAR differs slightly from the original SCF-exported one;
tighter diagonalization does not remove the 1.048 meV comparison discrepancy.
The existing validation threshold remains unchanged.

More seriously, for `sample_020`, learned-V/exact-D gives a frontier MAE of
13.519 meV relative to the tight exact-operator replay, whereas exact-V/learned-D
gives 2312.165 meV and learned-V/learned-D gives 2299.618 meV, with spurious
states and a negative sampled gap. Thus the PAW D prediction, not the smooth
potential alone, causes this failure. For `sample_023`, the corresponding FNO
errors are 6.889, 2.354 and 5.477 meV.

Two imported-operator diagnostic solves reached the numerical energy floor
under EDIFF=1e-11. Fresh-directory retries used the original EDIFF=1e-9 with
NELMIN=32; job 16258664 completed in 41 s. This changes neither the trained
models nor the 1 meV validation threshold. The combined, provenance-linked
results are retained in
`runs/vasp_mos2_operator_diagnostic_retry_v1/diagnostic_summary.json`.
The original MoS2 checkpoint remains unvalidated. The separate conditioned
readout repair and its validation are described below.

### Conditioned PAW readout repair

`vasp_paw_conditioning.py` and `vasp_paw_repair.py` implement a separate
head-only repair of the MoS2 models. The original checkpoints, spectra, and
failed validation records remain unchanged. The original MACE features and
potential decoder are frozen. Only the new species-specific equivariant PAW
heads are fitted, using the original 16/4/4 split.

The native augmentation matrix Q has nearly dependent repeated radial
channels. An absolute matrix-entry loss can hide errors in combinations of
those channels. For each species, diagonalize its geometry-independent Q and
form B = |Q|^(1/2) and W = B^(-1). The encoded target is Z = W (D - D_atomic) W^T.
A training-only rotationally invariant mean and species RMS normalize Z.
The head predicts the normalized Z; the physical correction is reconstructed
as B Z B^T in float64. Training labels are read from the original float64
exports, not the float32 caches. The scalar projection retains all same-l
radial couplings, so the centering and scaling preserve O(3) equivariance.
All channel directions are retained; a relative eigenvalue floor of 1e-8
only safeguards the coordinate transform.

Q is the augmentation matrix, **not the full positive PAW overlap operator S**.
This coordinate scaling is not an exact physical operator-error norm and
cannot by itself guarantee the absence of ghost states. Native diagonalization
remains necessary. On the original SiLU/FNO validation structures, the encoded
matrix RMS error is about 52.3 for S and 1.72 for Mo, although the ordinary D
RMSE is small. This identifies a poorly controlled part of the readout;
it does not identify a unique microscopic origin of every spurious state.

The repair first uses 4000 full-batch Adam updates on cached frozen MACE
features, with validation selection every 25 updates and equal weight per
species. The PAW head is linear in its trainable parameters when these features
are fixed, so the objective is quadratic. A subsequent L-BFGS refinement uses
at most 100 outer iterations of 20 inner iterations, strong-Wolfe line search,
and validation selection after every outer iteration. It retains the initial
checkpoint if refinement does not improve validation. No test labels enter
fitting or checkpoint selection. All five original
models receive the same repair protocol. The original failed test structures
have already been inspected, so this is a regression check on known failures,
not a new independent generalization benchmark.

New native band solves use three in-plane k points, 112 bands, EDIFF=1e-9,
and NELMIN=32. Validation structures are checked before new test predictions.
A predeclared spectral sanity check requires all calculated bands to agree
within 0.25 eV with the original SCF-exported operator and a positive sampled
gap. This is a failure detector, not an accuracy target or a substitute for
reporting errors. A separate same-operator native/export/replay check retains
the 1 meV threshold. The original SCF-export versus CHGCAR-reconstruction
comparison is also reported at its unchanged 1 meV threshold; it is not
silently replaced by the same-operator check.

The first launch (`vasp_mos2_paw_conditioned_v1`, job 16258672) stopped before
optimization because a geometry-only cache used a 1x1x1 dummy field, below the
input validator's minimum size. The corrected cache uses 2x2x2; no field values
enter the MACE graph. The replacement experiment is
`runs/vasp_mos2_paw_conditioned_v2` (job 16258673), with a frozen source
snapshot. All its validation spectra passed the failure detector, but the
SiLU/FNO frontier MAE was still 31.925 meV and the training objective was still
improving. Its queued test job 16258675 was canceled before execution.
The refinement is `runs/vasp_mos2_paw_conditioned_v3` (validation job 16258706);
its manifest hashes the parent checkpoints as well as the original field
models. Source tests
cover signed-Q scaling, rotation/inversion, precision in weak radial directions,
and preservation of the frozen field.

The refined validation set passed all five models and all same-operator replay
checks. The mean frontier MAEs are 19.162 meV (MACE), 19.494/23.785 meV
(pointwise SiLU/GELU), and 10.735/7.116 meV (FNO SiLU/GELU).
The maximum same-operator replay error is 0.001 meV.

The original evaluation launch `vasp_mos2_paw_conditioned_eval_v1`
(job 16258709) stopped at the frozen-potential reproducibility check, before
native test-band solves. Fresh inference processes allowed cuDNN TF32 whereas
training disabled it. Potential differences reached 1.77 meV for SiLU/FNO.
`vasp_train.predict` now explicitly disables both cuDNN and matrix-multiplication
TF32, matching the training precision. A regression test checks that policy
inside the correction's forward call. The evaluation was restarted in
`vasp_mos2_paw_conditioned_eval_v2` (job 16258712) using the identical saved
checkpoints and an unchanged 1e-5 eV field-reproducibility tolerance. The first
repeated SiLU/FNO field differs from its original prediction by at most
1.43e-6 eV, consistent with ordinary float32 GPU reductions. A separate
evaluation snapshot preserves the original training source and manifests.

A repaired checkpoint is loaded using
`mapho.vasp.checkpoints.load_repaired_model(repair_root, label, device)` and passed to
`mapho.vasp.train.predict`. The latter still receives only the geometry/atomic
baseline. Its predicted potential and D feed the existing `prepare_replay`
interface; the VASP executable and native projectors/Q are unchanged.

#### Completed repair results

Job 16258712 completed successfully in 8 min 15 s. Every repaired model passed
the predeclared spectral checks on all four existing test structures, following
the four-structure validation gate. All 112 bands were checked at the same
three in-plane k points. No observed spurious states remain in these spectra;
the largest band error among all five models/structures is 0.119 eV.
The frontier statistic uses four occupied and four empty bands.

Before/after MAEs below use exactly the same original SCF-exported reference
spectrum, with no fitted energy shift. They differ slightly from the older
native-CHGCAR-based table because that comparison used a slightly different
reference operator.

| MoS2 model | Original frontier MAE (meV) | Repaired frontier MAE (meV) |
| --- | ---: | ---: |
| MACE alone | 853.650 | 23.927 |
| Pointwise / SiLU | 760.869 | 25.406 |
| FNO / SiLU | 587.026 | 11.505 |
| Pointwise / GELU | 640.820 | 31.481 |
| FNO / GELU | 540.329 | 8.898 |

For the formerly failing `sample_020`, SiLU/FNO improves from 2299.618 to
15.729 meV frontier MAE. Its sampled gap changes from -0.260777 eV to
1.325318 eV, compared with the reference 1.316966 eV. All 112 predicted bands
at the checked k points are within 37.3 meV of the reference. On `sample_023`,
the local MACE error falls from about 959 to 10.5 meV. These corrections were
learned from training structures; test structures were not used for fitting
or checkpoint selection.

The smooth potential is preserved to within 1.91e-6 eV over all test grids.
The new D RMSE is about 2.3 meV for each variant. The identical native
operator export/replay test reproduces all test eigenvalues at their printed
precision (maximum difference zero; validation maximum 0.001 meV).
The separate original-SCF versus CHGCAR reconstruction discrepancy remains
1.048 meV on `sample_023`, so `original_reference_gate_passed` remains false.
The original 1 meV gate was not relaxed and the original failed results were
not overwritten. The repair addresses the spurious spectra; it does not
remove that distinct reference-reconstruction difference.

Final artifacts are in `runs/vasp_mos2_paw_conditioned_eval_v2`:
`comparison.csv`, `repair_audit.json`, `test_summary.json`, all predicted
potential/matrix files, and all native VASP inputs/outputs. Trained heads and
their frozen-base model contracts are in `runs/vasp_mos2_paw_conditioned_v3`.
The latter also contains `readout_audit.json`, quantifying the improvement in
both native and rescaled matrix coordinates on training and validation data.

For inference, load a repaired model rather than the original failed readout:

```python
from mapho.vasp.checkpoints import load_repaired_model as load_model
from mapho.vasp.train import predict
from mapho.vasp.paw import sha, prepare_replay

model = load_model("runs/vasp_mos2_paw_conditioned_v3", "silu_full", "cuda")
# atomic_h5 is the geometry-only VASP atomic-reference operator for a structure.
v, d = predict(model, {"baseline": atomic_h5, "baseline_sha256": sha(atomic_h5)}, "cuda")
prepare_replay(atomic_h5, output_directory, potential=v, d=d, kpoints=kpoints)
```

This is a regression evaluation on the previously inspected four-structure
test set, following validation-based refinement, not a fresh independent
benchmark. It establishes the removal of the observed failures over these
geometries and k points, not a general guarantee against ghost states or
transfer to other pseudopotentials, magnetic states, or geometries. The
coordinate scaling, float64 reconstruction, and optimizer refinement were
changed together; their separate contributions have not been ablated.

### Forces and finite-displacement phonons

The current imported-operator route is validated for eigenvalues, not forces.
VASP 6.6.1 explicitly skips `FORCE_AND_STRESS` when `INICHG=4`, the
potential-read mode used by `ICHARG=14`. The bridge does not alter this guard.
The existing distorted Si16 ML replays consequently contain zero force arrays
in `vasprun.xml`. These arrays must not be supplied to phonopy as physical
forces. A static calculation alone does not change this behavior.

Phonopy's displaced-supercell workflow can be used once consistent forces
are implemented and validated. This requires more than removing the guard:
the imported potential and PAW matrices need an appropriate energy/force
evaluation. Non-SCF forces are possible in principle; VASP's documented
`ICHARG=12` Harris–Foulkes route includes its own energy functional and force
corrections, unlike the current arbitrary-operator import. An alternative is
to initialize a subsequent SCF calculation from ML-generated electronic data
and use its converged native forces. Neither extension is implemented here.

### Primitive LiF and MoS2 band paths (2026-10-16)

`runs/vasp_binary_primitive_bands_v1` evaluates the frozen SiLU pointwise and
FNO checkpoints on the unstrained primitive parents of the training structures.
LiF uses the original LiF16 models; MoS2 uses the conditioned-v3 PAW heads with
the original field models frozen. No primitive reference labels are used for
prediction or model selection. The new driver is
`mapho.experiments.vasp.binary_primitive_bands`; plotting uses
`mapho.experiments.vasp.binary_primitive_plot`. All calculations and artifacts are in scratch.

The two-atom rocksalt cell has conventional a=4.03 A. The six-atom 2H bilayer has
in-plane a=3.18 A, Mo-plane separation 6.15 A, layer S-S thickness 3.19 A, and
20 A vacuum (c=29.34 A). These constants come directly from the dataset generator;
they are ideal, unrelaxed choices without an experimental citation in the
provenance. Primitive reduction explicitly checks reconstruction of the parent
supercell. Calculations use nonmagnetic PBE, 600 eV, no SOC or empirical dispersion.
Symmetry is used for primitive SCF sampling and disabled for the explicit band paths.

| Material | SCF meshes checked | Path | Points | Bands |
|---|---|---|---:|---:|
| LiF | 19x19x19 / 22x22x22 | X-Gamma-L | 81 | 16 |
| MoS2 bilayer | 20x20x1 / 23x23x1 | Gamma-M-K-Gamma | 73 | 40 |

The denser SCF result supplies the native ICHARG=11 reference. All 18 VASP
calculations finished with electronic-convergence certification and GPU
offloading. Changing the SCF mesh shifts frontier levels by at most 0.002 meV
(LiF) and 0.001 meV (MoS2). Native export/replay has zero frontier discrepancy at
printed precision and maximum all-band discrepancies of 0.005 and 0.077 meV.
The separate SCF-export versus CHGCAR-reconstruction frontier discrepancy is
0.001 meV for LiF and 0.290 meV for MoS2; it is not conflated with replay error.

| Material | Method | Path gap (eV) | Frontier MAE (meV) | Frontier maximum (meV) |
|---|---|---:|---:|---:|
| LiF | PBE reference | 9.075329 | 0 | 0 |
| LiF | MACE + pointwise | 9.011793 | 40.243 | 71.283 |
| LiF | MACE + FNO | 8.989321 | 34.101 | 57.859 |
| MoS2 bilayer | PBE reference | 1.236519 | 0 | 0 |
| MoS2 bilayer | MACE + pointwise | 1.239798 | 5.221 | 11.330 |
| MoS2 bilayer | MACE + FNO | 1.229774 | 12.115 | 24.404 |

Frontier errors include the top two occupied and bottom two unoccupied levels
at every path point, without a fitted energy shift. Gaps are sampled along the
specified paths, not a full Brillouin-zone minimization. FNO has lower frontier
MAE for LiF, while pointwise is better for this primitive MoS2 case. This is one
parent geometry per material and one training seed, not an aggregate ranking.
The repaired MoS2 models have positive gaps and no recurrence of the previously
observed spurious-state problem on these paths; this is not a general proof.

`figures/lif_mos2_primitive_bands.pdf` and the PNG preview compare native and
predicted dispersions with a common reference VBM zero. Lower panels show the
pointwise and FNO frontier errors. The 25,296-row `band_source_data.csv` contains
all levels, including those outside the declared display windows, and all
reference controls. The figure bundle retains its plotting source, caption,
metrics, input/checkpoint audit, and passing PDF text/collision checks.


### Joint fine-tuning after the MoS2 PAW repair (2026-10-16)

`runs/vasp_mos2_joint_conditioned_v1` contains joint fine-tuning of the repaired
SiLU pointwise and FNO models, followed by native VASP validation, test and
primitive band calculations. LiF retains its original jointly trained model.
The driver is `mapho.experiments.vasp.paw_joint`. Load the new models with
`mapho.vasp.checkpoints.load_joint_model`, using labels `silu_none` or `silu_full`.

Both losses now propagate through the shared MACE backbone. The potential
coefficients, offset, correction network and conditioned PAW heads are all
trainable. Replaced, unused original PAW heads remain frozen. Gradient tests
verify that the field and PAW losses each reach MACE; saved parameter differences
confirm every active group changed in both fitted checkpoints. The existing
frozen repair behavior remains the default of `ConditionedPAWModel`.

The original 16/4/4 train/validation/test split and training-only scales remain
fixed. The fit uses 300 full-batch Adam updates (16 structures per update), with
learning rates 3e-5 for MACE, 1e-4 for the field/correction and 3e-4 for PAW heads.
The objective combines normalized field MSE and equally weighted species errors
in conditioned PAW coordinates. PAW targets are encoded from float64 source
matrices before conversion to training precision. Validation includes the initial
checkpoint and selects epoch 300 for both models. Validation was still improving;
this budget does not establish convergence. Extra optimization and unfreezing
are changed together, so this comparison does not isolate their contributions.

| MoS2 evaluation | Model | Before frontier MAE (meV) | Joint frontier MAE (meV) |
|---|---|---:|---:|
| Validation supercells | Pointwise | 19.494 | 18.480 |
| Validation supercells | FNO | 10.735 | 8.405 |
| Held-out supercells | Pointwise | 25.406 | 23.835 |
| Held-out supercells | FNO | 11.505 | 10.077 |
| Primitive band path | Pointwise | 5.221 | 8.040 |
| Primitive band path | FNO | 12.115 | 4.068 |

Supercell frontier metrics include four occupied and four unoccupied bands at
three k points for each of four structures. Primitive metrics use two occupied
and two unoccupied bands at each of 73 path points. They are distinct evaluation
sets and band windows. No model-specific energy shifts are fitted. Both held-out
averages improve, but the pointwise primitive error increases. The joint FNO
primitive path gap is 1.228259 eV, pointwise 1.239803 eV, and PBE 1.236519 eV.
A lower frontier MAE does not necessarily imply a more accurate gap.

All 18 new GPU VASP calculations converged and the validation/test spectral
sanity checks passed. Parent references were reused with verified hashes. Inputs,
checkpoints and the immutable training source snapshot are audited. The same
previously examined test structures were evaluated without fitting or selecting
on their labels; this remains one seed and is not an independent benchmark.

The figure bundle includes `lif_mos2_primitive_bands.pdf`, its PNG preview,
`joint_comparison.csv`, `metrics.csv`, 31,136 source energy rows including both
starting and joint models, a caption, analysis/plotting sources and passing
integrity, text and collision audits. The source snapshot used for fitting is
preserved; a postprocessing-only revision recomputes primitive field errors and
preserves the original Slurm job ID (16258909).
