# Controlled spectral-mixing experiment

This experiment asks whether spatial Fourier mixing improves a restricted
exchange-gate correction when its local path, inputs, and training protocol are
controlled. It uses the existing two-atom LiF PBE0 dataset. It does not test
long-range transfer across cell sizes or replace the exact Fock operator.

The older width-four FNO versus width-sixteen pointwise comparison confounded
spectral mixing with local nonlinear capacity. Its 36-epoch fits also selected
their final epochs. Those results remain exploratory measurements at a short
training budget.

## Architectures

All new corrections use three inputs, a width-sixteen lift, two width-sixteen
layers, and a scalar readout. The readout starts at zero. At each layer the
common pointwise affine map is followed by SiLU after adding the optional
branch.

| Mode | Added branch in each layer | Field SiLU stages on longest path | Correction parameters |
| --- | --- | ---: | ---: |
| `none` | None: the FNO's spectral branch is removed | 2 | 625 |
| `full` | Original metric radial Fourier filter | 2 | 9,393 |
| `local` | Pointwise 16–132–16 MLP with SiLU | 4 | 9,369 |
| `mean` | Only the zero-wavevector contribution of the Fourier filter | 2 | 9,393 |

`full` versus `none` isolates the effect of adding the spectral branch to an
otherwise identical local path. `local` tests whether additional local capacity
can explain the gain. `mean` tests whether a whole-cell average is sufficient.
Its nominal parameter count matches `full`, but evaluation of its radial network
at zero leaves some weights inactive and provides fewer effective degrees of
freedom. It is a mechanism control, not an effective-capacity match.

The local control includes an extra nonlinear operation inside each added
branch. It approximately matches the total parameter count, but does not match
the nonlinear depth of the full FNO on the field inputs. The FNO radial MLP's
SiLU acts on wavevector magnitude, not on the incoming field, so it does not add
an equivalent field activation. The stage counts above exclude the shared MACE
backbone and final gate transformation; the lift and readout are linear. All
modes have exactly matching initial values for the shared pointwise parameters;
this is checked by hashes before held-out evaluation.

## Training and selection

- Common MACE initialization: the existing LiF PBE0 MACE-only seed-31
  checkpoint. This is shared by all variants and all five seeds; the seeds do
  not represent independent MACE pretraining runs.
- Frozen phase: reuse the common MACE field and train the correction only.
  Joint phase: also update the MACE backbone and field decoder.
- Seeds: 31, 47, 61, 79, 97. The data order is paired across variants.
- Inputs: common MACE field, cube root of nonnegative PBE density, and PBE Hxc.
  Centering and scaling use training structures only and are fixed for all
  variants, including joint training.
- Keep the pretrained MACE neighbor normalization at 16. Changing it in this
  experiment would also change the shared starting model.
- Each epoch uses all eight training structures. Each structure update uses
  all six permitted bands at one k-point; the six permitted k-points cycle
  equally over six epochs. This is not an all-k-points-per-step optimizer.
- Adam: correction learning rate 0.001, joint MACE learning rate 0.0001;
  gradient norm clipped to one.
- Validate every six epochs on the two validation structures, all six
  permitted k-points, and all six permitted bands. Select the minimum validation
  RMSE, allowing the unmodified initial model to win.
- Reduce learning rate by half after five validation checks without an
  improvement greater than 0.01 meV. Stop after at least 120 epochs and 60 epochs
  without such an improvement; maximum 720 epochs. A fit reaching this maximum
  must not be described as having reached a validation plateau.
- Save optimizer, scheduler, best model, and data-order state every validation
  check. A resumed fit has the same trajectory as an uninterrupted fit in the
  CPU regression test.

The correction networks and exchange FFTs run on CUDA. The small MACE graph runs
on CPU with differentiable coefficient transfers to CUDA in joint fits. The
small-cell probe basis is cached with the existing 256-MiB per-structure bound;
this is not the production large-cell streaming memory path.

## Evaluation and interpretation

Only after all requested fits complete, evaluate the six test structures with
all eight k-points and eight bands. Report exchange-action RMSE, separate
upper-valence and unoccupied errors, unseen k-points and bands, and Ritz errors.
The Ritz calculation retains the exact PBE0 local Hamiltonian and an eight-state
reference subspace. It is not a complete predicted PBE0 band calculation.

Include the uncorrected PBE-orbital exchange operator, the common pretrained
MACE gate, and an analytically fitted constant gate. The constant gate minimizes
training exchange-action MSE within the same gate bounds; it retunes a single
exchange multiplier without using atomic features.

Seed differences are paired descriptive comparisons. Six related distorted
structures and five optimization seeds are not independent material families.
The held-out structures were examined in earlier experiments, so this remains
an exploratory comparison, not a fresh confirmatory test set. Even `none` uses
PBE density and Hxc fields containing nonlocal information, and the MACE decoder
includes a global offset. Winning here would not demonstrate learning
long-range exchange or extrapolation to low nonzero wavevectors.

## Artifacts and execution

Runner: `python -m mapho.exchange_ablation_train`; model definitions are in
`src/mapho/exchange_ablation.py`. Existing production field and exchange model
definitions are unchanged.

Batch script: `runs/lif_pbe0_spectral_ablation_v1/main.sbatch`.
Output directory: `runs/lif_pbe0_spectral_ablation_v1/experiment/`.
Its protocol records source, manifest, and common-checkpoint hashes. Each fit
records the selected epoch, stopping reason, parameter changes, initialization
hash, timing, and CUDA peak allocation. `summary.json` and `completion.json`
are written only after held-out evaluation completes.

## Completed run and audit (2026-10-15)

Job 16242451 completed all 40 fits on an A100 in 46 minutes 49 seconds. The
following test exchange-action RMSEs are in meV, reported as mean ± sample
standard deviation across the five correction seeds. Lower is better.

| Correction | Frozen MACE | Joint MACE |
| --- | ---: | ---: |
| Spectral off (`none`) | 13.904 ± 0.275 | 13.853 ± 0.081 |
| Full FNO (`full`) | 13.663 ± 0.031 | 13.758 ± 0.014 |
| Extra pointwise MLP (`local`) | 13.762 ± 0.115 | 13.728 ± 0.020 |
| Cell average only (`mean`) | 13.802 ± 0.106 | 13.866 ± 0.116 |

The full FNO improves on the same-depth spectral-off control in all five seeds
in each phase, by a mean 0.240 meV with frozen MACE and 0.095 meV with joint MACE.
Against the larger pointwise control, its frozen mean advantage is 0.098 meV,
but only three of five paired seeds favor it (median advantage 0.011 meV).
With joint training the larger pointwise control wins all five pairs by a mean
0.030 meV. This latter comparison also changes nonlinear depth, as described
above. These results show a small benefit from adding Fourier mixing to the
fixed local path, without establishing an advantage over additional local
capacity or demonstrating long-range generalization.

The uncorrected PBE-orbital exchange reference has action RMSE 28.078 meV,
the fitted constant gate 25.924 meV, and the common pretrained MACE gate
16.455 meV. A constant rescaling therefore explains only a small part of the
improvement. The pretrained MACE baseline has a shorter training budget, and
there is no pointwise-without-MACE control here; these numbers do not isolate
the value of the MACE backbone or attribute all improvement to Fourier mixing.

The corresponding Ritz eigenvalue MAEs are also in meV:

| Correction | Frozen MACE | Joint MACE |
| --- | ---: | ---: |
| Spectral off (`none`) | 2.894 ± 0.269 | 2.803 ± 0.128 |
| Full FNO (`full`) | 2.897 ± 0.095 | 2.855 ± 0.091 |
| Extra pointwise MLP (`local`) | 2.919 ± 0.146 | 2.637 ± 0.117 |
| Cell average only (`mean`) | 2.689 ± 0.123 | 2.652 ± 0.051 |

The action advantage does not give a consistent Ritz advantage. These Ritz
diagnostics retain the exact PBE0 local Hamiltonian and an eight-state reference
subspace, rather than predicting a full PBE0 band structure.

An artifact audit checked all 40 checkpoints and independently recomputed the
metrics from all 16,512 saved evaluation rows. It verified source and data
hashes, split separation, training-only normalization, checkpoint selection,
finite parameters, paired common initialization, unchanged frozen MACE weights,
and changed joint MACE weights. It did not re-execute model predictions.
Thirty-eight fits met the configured validation-plateau stopping criterion;
`frozen_none_seed79` and `frozen_mean_seed31` reached the 720-epoch cap while
still improving. A plateau criterion is not proof of global convergence.

Detailed results: [summary.json](../runs/lif_pbe0_spectral_ablation_v1/experiment/summary.json).
Verification: [audit.json](../runs/lif_pbe0_spectral_ablation_v1/experiment/audit.json).
The `runs` directory now points to `/gpfs/scratch/acad/htbase/wchen/mapho/runs`.

## Longitudinal supercell references

The next reference batch is prepared by `mapho.qe.supercell_reference` under
`runs/lif_supercells_v1`. It retains the Li/F SG15 pseudopotentials (including
Li semicore states), 100/400 Ry cutoffs, and PBE0 with 25% unscreened exchange,
Gygi–Baldereschi divergence treatment, and no Gamma extrapolation. Each geometry
has a PBE and a PBE0 SCF calculation. The PBE0 calculation starts from the
converged PBE save for the same geometry.

| Conventional-cell replication | Atoms | Long dimension (A) | Smallest nonzero longitudinal q (inverse A) |
| --- | ---: | ---: | ---: |
| 4 x 1 x 1 | 32 | 16.12 | 0.390 |
| 6 x 1 x 1 | 48 | 24.18 | 0.260 |
| 8 x 1 x 1 | 64 | 32.24 | 0.195 |

The transverse dimensions remain 4.03 A: this is an initial test of variation
along one direction, not a fully three-dimensional large-cell benchmark. Each
size has pristine, longitudinal optical, transverse optical, mixed optical and
acoustic, and compact localized optical perturbations. Displacements are at
most 0.06 A in magnitude. The largest localized case leaves the atomic
environment within 12 A of the x=0 plane unchanged, providing a useful distant
perturbation diagnostic. One additional 64-atom case exactly repeats the
32-atom longitudinal pattern twice; it is a folding/sampling control rather
than an independent geometry. There are 16 supercell cases in total.

The 1 x 2 x 2 k and exchange-q meshes are feasibility settings. Their convergence
is not established, and longitudinal sampling changes with cell length.
Pristine and repeated-pattern controls help identify these effects; a later
accuracy comparison must assess sampling error before attributing differences
to model range. The current PBE-field inputs already carry global information,
which remains a separate limitation of a local-versus-spectral interpretation.

An isolated MPI CPU QE build is made from local commit
`aba99b18c9edac5380ab540650704cc9c373aece` (8.0dev), with compiler and library
settings in `configure.log` and source provenance in `build_provenance.json`.
The existing serial QEpy environment and its archived results are unchanged.
A primitive-cell comparison requires PBE and PBE0 eigenvalues to agree with
the archived QEpy 7.2 calculations within 1 meV before the new batch proceeds.
This check passed with maximum differences of 0.0394 meV (PBE) and 0.0335 meV
(PBE0).
A 32-atom smoke calculation then tests the production MPI layout. The remaining
cases use a Slurm array, at most four concurrent debug nodes and 128 MPI ranks
per case. `submission.json` records current job IDs and superseded attempts.
The smoke calculation (job 16250720) passed both functionals in 344 seconds.
Array 16250796 runs the other 15 cases largest first; at launch, four 64-atom
cases completed PBE and proceeded to PBE0 on four nodes. Each job has a two-hour
wall-time limit. These are CPU MPI calculations, not GPU-accelerated QE runs.

The native XML, charge density, and wavefunctions are retained on scratch.
The launcher checks charge convergence, PBE0 outer exchange convergence,
insulating gaps, and input/pseudopotential/executable checksums. Each completed
case receives `completion.json`; failures are recorded separately. Initial
primitive and 32-atom checks failed with distributed subspace diagonalization
(`-ndiag 4`). Both 128-rank PBE tests with one and four band groups passed when
the subspace eigensolve used one rank. The production layout therefore uses
four k-point pools, one band group for PBE, four band groups for PBE0, and
`-ndiag 1`. The bulk FFT and exchange work remain distributed. Failed attempts,
the original source snapshot and manifest, and the passing layout tests are
preserved under `attempts/` and in `layout_validation.json`.

This batch produces native reference saves and checked eigenvalue reports.
It does not yet export generalized exchange-action training NPZs, train new
models, or establish an FNO advantage. The eight-band pilot exporter is not
used for these 88-, 128-, and 168-band calculations.

## Automatic k-point proposals and convergence checks

`mapho.kpoints.mesh_from_spacing(atoms, spacing)` uses pymatgen's
`Kpoints.automatic_density_by_lengths`, with a reciprocal-metric correction,
so that each division satisfies `|b_i| / n_i <= spacing_i`. Reciprocal vectors
include `2*pi`, and spacing is in inverse Angstrom. A scalar or three directional
spacings are accepted. The metric correction also handles skewed cells. The
mesh is Gamma centred with zero QE offsets. ASE writes QE input files; a VASP
KPOINTS file is not passed to QE. The optional `references` dependency installs
pymatgen. Geometry-based proposals do not by themselves establish convergence.

The first convergence study is `runs/lif_kpoint_convergence_v1`. It covers the
pristine and longitudinally distorted 32-, 48-, and 64-atom cells. Transverse
spacings of 0.80, 0.55, and 0.40 inverse Angstrom give `1x2x2`, `1x3x3`, and
`1x4x4`; a separate longitudinal-spacing refinement gives `2x4x4`. Every mesh
has paired PBE/PBE0 inputs. PBE0 uses the same divisions for exchange q sampling.
All cutoffs, pseudopotentials, occupations, exact-exchange settings, geometries,
and potential conventions are held fixed. The manifest records pymatgen's
version, requested and actual spacing, and source/input/binary checksums.

```bash
.venv/bin/python -m mapho.qe.kpoint_convergence prepare \
  --root runs/lif_kpoint_convergence_v1 \
  --baseline runs/lif_supercells_v1 \
  --spacings 0.80 0.55 0.40 --tolerance-ev 0.001
.venv/bin/python -m mapho.qe.kpoint_convergence summarize \
  --root runs/lif_kpoint_convergence_v1
```

The preparation command requires a new output directory. The summary requires
both successive transverse comparisons and the longitudinal doubling check to
pass for both functionals. The initial thresholds are 1 meV for the maximum
change in four occupied and four empty frontier levels at common k points,
1 meV for the sampled gap, and 1 meV/atom for total energy. Gamma must be common;
no fitted eigenvalue shift is subtracted. Different meshes sample different
extrema, so the sampled gap is checked separately from common-k levels. Failure
to pass the prepared ladder means that a denser study is needed, not that its
last mesh is converged. These criteria do not establish cutoff convergence,
field convergence, or sub-meV accuracy for subsequent model comparisons.

Production array 16251238 follows restart-validation job 16251237. At most four
128-rank CPU debug nodes run simultaneously. Each geometry completes its PBE
ladder before its PBE0 ladder. Completed coarse references can be reused only
after matching physical inputs and checking their saved XML and SCF status.
Native clean time-limit stops can continue with the same MPI layout, up to four
two-hour allocations per geometry. Hard timeouts, numerical failures, missing
restart files, and exhausted allocation budgets remain unverified. Native
fields still require a convergence check before field training: `training_ready`
remains false even if all spectral criteria pass. No training is submitted.

The eleven unstarted tasks from array 16250796 were deferred to prioritize this
study; their inputs remain intact and their IDs are recorded in
`lif_supercells_v1/deferred_for_convergence.json`. The four already-running
coarse references were left to finish. Runtime source snapshots isolate the
new study from subsequent code edits. Inputs, native saves, logs, summaries,
and submission records remain on scratch.

### Recovery and longer hybrid allocations

The initial convergence study completed 20/24 PBE and 9/24 PBE0 references
before recovery. Four hybrid calculations ended without a certified checkpoint
at the two-hour debug limit. A fifth series stopped because the extxyz geometry
round trip changed Cartesian coordinates by at most 4.7e-9 A and an exact text
comparison rejected reuse. These failures were not counted as completed data.

`mapho.qe.kpoint_recovery` prepares an isolated recovery study under
`runs/lif_kpoint_recovery_v1`, retaining the existing inputs, incomplete saves,
and runtime snapshots. Verified completed functional directories are linked,
not recalculated. Reuse now tolerates at most 1e-8 A of Cartesian coordinate
serialization roundoff while requiring identical species, constraints, other
cards, and physical namelist settings. Newly prepared studies read coordinates
from the QE input rather than the lower-precision extxyz representation. Launcher
timeouts now write a failed state rather than leaving a stale running state.
Tests cover roundoff acceptance, rejection of physical changes, and timeout
failure recording; 15 targeted tests passed.

With the user's approval, long PBE0 jobs move to the batch partition. Initial
array 16255538 contains seven jobs, with two nodes, 256 ranks, 220 GiB/node,
a 24-hour limit, and at most four simultaneous jobs. These recover the four
interrupted 32-/48-atom calculations, add the missing 64-atom longitudinal
`1x3x3` reference, and refine the two 32-atom geometries to `1x6x6`. Corresponding
PBE inputs use the same k meshes and the hybrid exchange q meshes match k.
Five short PBE checks run on debug in array 16255537. The other dense hybrid
inputs are preserved for the next stage rather than all being launched at once.

Two-node validation job 16255536 completed in 230 seconds. Its 32-atom PBE0
spectrum agrees with the completed 128-rank result to 4.4e-11 eV, passing the
1 meV gate for launching the batch jobs. This tests MPI layout consistency;
it does not establish k-point or cutoff convergence. All physical settings,
including ACE, unscreened exchange, and the divergence convention, are retained.

The already-active 64-atom pristine debug job 16255015 was left to finish its
allocation. Its continuation script was archived and routed to batch, keeping
128 ranks for a native checkpoint restart. A guard job covers an unclean stop:
only a certified checkpoint is resumed; otherwise a fresh SCF uses the verified
PBE save. This continuation finishes `1x3x3` only, avoiding an automatic jump to
denser meshes with untested memory requirements. Submission and script hashes
are recorded in the recovery study's `submission.json`.

`summary.json` retains the original convergence criteria. The added `1x4x4` to
`1x6x6` comparisons are reported separately in `refinement_summary.json`; one
passing comparison is insufficient for final mesh selection. In particular,
changes in a gap sampled on different k meshes mix SCF convergence with changes
in sampled band extrema. Common-k frontier-level and total-energy changes must
also be inspected. The completed 32-atom `1x3x3` to `1x4x4` PBE0 comparison shifts
common-k frontier levels by about 3 meV, so finer SCF sampling is needed even
apart from the sampled-gap criterion. Native field convergence, additional
geometries, training exports, and model training remain outstanding.
