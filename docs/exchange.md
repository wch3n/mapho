# Hybrid-exchange pilots

This experiment tests whether a Fourier neural operator helps predict a
nonlocal HSE06 exchange correction. It uses the installed MACE implementation
and the existing metric-aware FNO layers. It is separate from the PBE scalar
potential backend and does not extend that backend's supported functional list.
An extension below tests unscreened PBE0 exchange with the same LiF structures.

## Target and reference

The reference electronic input is a **converged PBE calculation**, including
all occupied orbitals per spin at every point of a full 2x2x2 k mesh (four
for the silicon pilot, five for LiF with Li 1s states in valence).
This electronic calculation is required at inference. The model does not
predict a hybrid Hamiltonian from atomic coordinates alone.

For each Bloch vector k, the implementation applies the screened Fock operator
using the physical momentum transfer G+k-q, all occupied q states, and the
analytic Fourier kernel

\[
w_\mu(Q)=4\pi[1-\exp(-Q^2/(4\mu^2))]/Q^2,
\quad w_\mu(0)=\pi/\mu^2.
\]

The exchange fraction is 0.25 and mu is 0.106 bohr^-1. Orbitals use an
orthonormal discrete FFT convention. The implementation accounts explicitly
for volume, q weights, Ry-to-eV conversion and same-spin occupations. No
dense plane-wave Hamiltonian or six-dimensional kernel is stored.

The learned operator is restricted to

\[
K_{\mathrm{ML}}(k)=P_k M_g K[\gamma_{\mathrm{PBE}}](k)M_gP_k,
\qquad g(r)=\exp[0.1\tanh f_\theta(r)],
\]

where P_k projects onto the plane-wave basis at k. M_g is multiplication by
a real field. The structure/reference-field predictor can be nonlinear;
application to a trial wavefunction is linear, Hermitian and negative
semidefinite. Rotations among equally occupied reference orbitals leave
the operator unchanged. The modulated density matrix is not constrained to
remain an idempotent projector or preserve its trace; this is an operator
surrogate, not a self-consistent density-matrix model or energy functional.

This ansatz preserves the Fourier convolution explicitly. FNO predicts a
modulation field, rather than replacing Coulomb convolution or learning an
arbitrary two-position exchange kernel. It does not reduce the number of
occupied-state FFT convolutions. Its ability to improve accuracy and its
application speed are separate questions.

## Reference calculations

`mapho.qe.exchange` runs isolated serial QEpy 7.2.0 workers, with f90wrap
0.2.14. The fixed pilot uses nonmagnetic two-atom Si cells, eight bands,
40/160 Ry cutoffs, fixed occupations, and the original SG15 PBE NC
pseudopotential. The HSE calculation deliberately uses that same PBE
pseudopotential. `exxdiv_treatment='none'` and
`x_gamma_extrapolation=False` specify a finite screened zero-mode convention
that is reproduced by the independent FFT operator. The 2x2x2 exchange mesh
is a pilot setting, not a converged bulk band-gap calculation.

The exporter calls QE's native exchange and ACE routines through a private
GNU Fortran ABI bridge. This interface is specific to the installed QEpy
wheel; it is not a portable public QE API. It checks both charge convergence
and the outer exchange convergence, verifies orbital normalization and
occupations, and requires native ACE to reproduce its construction states.
Each run records inputs, pseudopotential hash, conversion constants, exchange
error, timing and native labels. The ordinary PBE field files are untouched.

The exporter also accepts multiple species and explicit cutoffs. The neutral
electron count is computed from the UPF valences and checked against QE's
occupations. The eight-band pilot supports one to six occupied states per
spin, with at least two empty bands. The trainer reads the species table from
the dataset manifest and the occupied-state count from each reference report;
legacy silicon manifests remain readable.

## Comparison protocol

The pilot takes the first eight training, two validation and all six test
structures from the existing, fixed silicon split. No split is chosen using
exchange accuracy. It compares:

- Screened exchange evaluated from the PBE occupied orbitals without ML.
- A jointly trained MACE coefficient predictor.
- MACE plus a pointwise neural network with a similar parameter count.
- Jointly trained MACE plus FNO.
- Native HSE exact exchange and native HSE ACE.

Both trainable branches update in the combined models. All variants start
from the identity gate, use the same data/order per seed, and receive the
same number of optimizer updates. Seeds are 31 and 47; each run has 36
epochs. Each update uses two bands at one k point, cycling through all six
training k points and six training bands every 18 epochs. Validation uses
the two held-out validation structures, k indices 0 and 3 and bands 0--5;
the best validation checkpoint includes epoch zero as a candidate.

HSE exchange-action labels at k indices 6 and 7 and band indices 6 and 7
are excluded from fitting and checkpoint selection. All test structures are
held out. **The reference PBE density matrix still includes all eight q
points**, as required to construct exchange. Thus the held-out-k experiment
tests unseen target actions on the integration mesh; it is not a new
off-mesh band-path calculation.

Action error is the norm of `(K_prediction-K_HSE) psi` for a normalized
state, in eV. Reported aggregate RMSE is the square root of the mean squared
action norm, not a grid-point RMSE. Separate results cover occupied states,
unoccupied states, withheld k points and withheld bands. Two random probe
states per k, orthogonal to all eight HSE eigenstates, test behavior outside
that subspace.

Native ACE is built from all eight HSE bands. Its accuracy on those states
is a construction property, not a learned generalization result. Accuracy
on orthogonal probes is reported separately. Native application timings
exclude the preceding HSE SCF and ACE construction, which are separately
recorded. ML action timings exclude the PBE SCF and coefficient prediction;
the coefficient prediction time is reported separately. This pilot is not
an end-to-end hybrid speed benchmark.
Native QE arithmetic is complex128; learning and the timed PyTorch operator
use complex64. The independent interface check uses complex128. Timings on
the shared CPU host are indicative, not a controlled equal-precision speed
comparison.

Eight-state Ritz eigenvalue diagnostics replace exchange within the exact
HSE eigensubspace while retaining the exact HSE local Hamiltonian. They
isolate the spectral effect of exchange error. They are **not** full
plane-wave diagonalizations or an end-to-end predicted hybrid band structure.

## Reproduction

From the project directory, in fresh output directories:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-qe/bin/python -m mapho.qe.exchange dataset \
  --source runs/si_fields_v2 --output runs/hse_dataset \
  --pseudo runs/qe_si_roundtrip/Si_ONCV_PBE-1.2.upf

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m mapho.exchange_train --dataset runs/hse_dataset \
  --output runs/hse_fit --epochs 36 --seeds 31 47
```

The exporter supports `--resume` only for matching requests/protocols.
`--train-only` permits fitting once all training/validation workers have
completed, without reading any test data. `--checkpoints PATH...` evaluates
existing models against a completed dataset. Checkpoint metadata binds the
model to the dataset-manifest hash.

The workspace experiment is under `runs/hse_exchange_pilot_v1`. Its launcher
runs the two independent seeds on two CPU processes and evaluates all six
models together. `evaluation/comparison.json` and its per-state rows are the
primary results; training histories and parameter-change counts accompany
each checkpoint.

## Completed silicon pilot

Results below use six held-out structures, eight k points and eight bands
per structure. ML entries are arithmetic means of the aggregate RMSE from
the two seeds. A range is the two observed seed values, not a confidence
interval. All models received 288 optimizer updates per seed, with the best
validation checkpoint at epoch 36.

| Predictor | Action RMSE (meV) | Seed range (meV) | Eight-state Ritz MAE (meV) |
|---|---:|---:|---:|
| PBE occupied-state reference | 18.138 | — | 6.233 |
| MACE | 11.657 | 11.577–11.737 | 3.069 |
| MACE + pointwise control | 11.174 | 10.963–11.385 | 2.891 |
| MACE + FNO | 11.476 | 11.216–11.736 | 3.040 |

FNO reduces aggregate action error by 1.56% relative to MACE alone and by
36.73% relative to the PBE reference. The similarly sized pointwise control
outperforms FNO for every test structure in both seeds. Thus this experiment
does not establish an advantage from global Fourier mixing. It tests a
restricted common modulation field on a narrow family of two-atom cells;
it does not settle the usefulness of FNO for a more general nonlocal kernel,
larger cells, other chemistries or longer training.

| Predictor | Withheld target k points: action RMSE (meV) | Withheld bands 7–8: action RMSE (meV) |
|---|---:|---:|
| PBE reference | 18.329 | 9.648 |
| MACE | 11.964 | 6.191 |
| MACE + pointwise control | 11.466 | 5.617 |
| MACE + FNO | 11.779 | 6.025 |

The independent complex128 FFT implementation matches native HSE exchange
to a maximum action error of 1.51e-12 eV over all eight k points of the first
training structure. Native ACE reproduces its eight construction bands to
numerical precision. On the orthogonal random probes, its rank-eight
approximation has 133.56 meV action RMSE, versus 1.01 meV for MACE–FNO's
uncompressed reference-based operator. This is the expected limitation of a
fixed ACE subspace, not evidence that ML replaces ACE's numerical role.

Median application time for eight states at one k point is about 0.295 ms
for native ACE, 557 ms for native uncompressed exchange and 132 ms for the
PyTorch operator. MACE–FNO coefficient prediction adds about 162 ms once per
structure. Native ACE construction takes about 4.45 s per structure. The
reference PBE and HSE SCFs take about 1.98 and 33.20 s, respectively. These
components cannot be combined into an end-to-end hybrid speedup: the pilot
does not predict the hybrid local potential, and the implementations use
different arithmetic precision.

Both MACE–FNO runs changed 27 MACE backbone tensors, four field-head tensors
and 16 FNO tensors. Trainable parameter counts are 8,389 (MACE), 9,014
(pointwise) and 9,058 (FNO). Peak training-process memory is about 1.24 GiB,
including cached data. The maximum Hermiticity discrepancy in the evaluated
complex64 exchange matrices is 1.06e-6 eV. The full test suite passes 69 tests.

The machine-readable report is
[`summary.json`](../runs/hse_exchange_pilot_v1/summary.json), with full
[per-state results](../runs/hse_exchange_pilot_v1/evaluation/evaluation_rows.csv).

## LiF comparison protocol

The ionic-material comparison uses a two-atom rock-salt LiF primitive cell
with a chosen parent lattice constant of 4.03 angstrom. It uses the same
strain/displacement ranges, geometry seed 29, split sizes (8/2/6), eight k
points, eight bands, training seeds (31/47), 36 epochs and optimizer settings
as the silicon experiment. Each architecture is trained from scratch on LiF;
this is not a silicon-to-LiF transfer experiment. Both MACE and the correction
branch update in the combined models.

The SG15 PBE scalar NC UPFs have three valence electrons for Li and seven for
F. Li 1s states are therefore included: the operator sums **five occupied
states per spin** at every q point. In addition to all-eight-band errors,
results separate the three upper occupied bands (indices 2--4), three empty
bands (5--7), and the two deep occupied bands (0--1). Training on indices
0--5 therefore includes one empty band for LiF; the two other empty bands
remain withheld. The target-label and PBE-input distinctions of the silicon
experiment still apply.

A pristine-cell 80/320 versus 100/400 Ry check preceded dataset generation.
Increasing the wavefunction cutoff changes the HSE sampled gap by 1.202 meV,
the largest VBM-aligned eigenvalue by 7.411 meV, the exchange expectation by
at most 1.128 meV, and the unlearned reference action RMSE by 0.0852 meV.
Both independent FFT exchange calculations agree with native QE below
8.43e-13 eV. The pilot uses 100/400 Ry. This cutoff sensitivity check does
not establish bulk convergence of the 2x2x2 k mesh.

The workspace run is `runs/lif_hse_exchange_v1`; it records the downloaded
UPF source URLs and SHA-256 hashes, fixed geometry manifest, cutoff check,
model histories and the experiment protocol. To generate a fresh dataset:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-qe/bin/python -m mapho.qe.exchange lif-dataset \
  --output runs/lif_hse_dataset --ecutwfc 100 --workers 2 \
  --li-pseudo runs/lif_hse_exchange_v1/pseudos/Li_ONCV_PBE-1.2.upf \
  --f-pseudo runs/lif_hse_exchange_v1/pseudos/F_ONCV_PBE-1.2.upf

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m mapho.exchange_train --dataset runs/lif_hse_dataset \
  --output runs/lif_hse_fit --epochs 36 --seeds 31 47
```

Two isolated serial QE workers export independent labels concurrently. The
existing single-silicon CLI and reference files remain usable.

### Completed LiF results

These are mean action RMSEs over two independently trained seeds, evaluated
on six unseen structures, eight k points and eight bands. The edge group
contains the three upper occupied and three empty bands. Units are meV;
these are exchange-action errors, not band-gap errors.

| Predictor | All eight bands | Seed range | Six edge bands | Three empty bands |
|---|---:|---:|---:|---:|
| PBE occupied-state reference | 30.568 | — | 31.957 | 28.162 |
| MACE | 18.136 | 17.596–18.677 | 18.868 | 15.480 |
| MACE + pointwise control | **15.675** | 15.596–15.754 | **16.090** | **9.946** |
| MACE + FNO | 16.330 | 16.154–16.505 | 17.231 | 11.195 |

FNO improves on MACE alone by 9.96% in aggregate action RMSE, a larger
improvement than the silicon pilot's 1.56%. Nevertheless, the pointwise
control has 4.01% lower error than FNO. It wins on every test structure in
both seeds for both the all-band and six-edge-band metrics. Moving to this
ionic material therefore does not establish an FNO-specific advantage in
the current restricted modulation ansatz. No hyperparameters were retuned
for LiF. This comparison changes material, cell size and pseudopotentials;
it does not isolate ionicity as the cause of a numerical difference.

For withheld target k points, MACE, pointwise and FNO errors are 18.134,
15.542 and 16.257 meV. For the two withheld empty bands they are 11.226,
6.317 and 7.592 meV. Eight-state Ritz eigenvalue MAEs are 5.190, 2.991 and
3.243 meV, respectively, versus 10.122 meV for the unlearned reference.
These Ritz diagnostics retain the exact HSE local Hamiltonian and HSE
eigensubspace; they are not a complete predicted hybrid band structure.
FNO has a slightly lower upper-valence Ritz MAE than the pointwise control
(3.354 versus 3.717 meV), despite its larger upper-valence action error.
Thus the aggregate action ranking is not a claim that every spectral
diagnostic favors the same model.

All six fits selected epoch 36 using validation data. Parameter counts are
8,941 (MACE), 9,566 (pointwise) and 9,610 (FNO). Both combined models update
27 backbone tensors and four field-head tensors; the FNO branch updates
16 tensors, and the pointwise branch updates eight. Each fitted model
received 288 optimizer updates. All reference cells use a 36x36x36 FFT grid.

The completion audit confirms 16 converged PBE/HSE pairs, disjoint splits,
six fitted checkpoints bound to the dataset hash, and 3,072 finite evaluation
rows. Maximum outer exchange residual is 8.04e-12 Ry; native ACE agrees with
its construction states within 4.43e-14 eV per coefficient. Independent FFT
exchange agrees with QE within 4.55e-15 eV on the first distorted training
cell. The largest complex64 exchange-matrix Hermiticity discrepancy is
3.13e-6 eV. The code suite passes 71 tests, including the new semicore-count,
species/hash-validation and legacy-silicon checks.

See the [LiF summary](../runs/lif_hse_exchange_v1/summary.json),
[per-state CSV](../runs/lif_hse_exchange_v1/evaluation/evaluation_rows.csv),
and [completion audit](../runs/lif_hse_exchange_v1/audit.json).

## Unscreened PBE0 extension

The PBE0 comparison retains 25% exact exchange with the full Coulomb kernel
`4*pi/|G+k-q|^2`. It has no erfc screening or real-space Coulomb cutoff.
The isolated zero-momentum term is handled by QE's Gygi--Baldereschi Gaussian
auxiliary-function correction, with `x_gamma_extrapolation=False`. This is
a quadrature treatment of the integrable singularity, not an interaction
range cutoff. The HSE finite zero-mode formula must not be evaluated at
screening parameter zero.

`FockExchange` implements both screened and unscreened kernels. Its original
`ScreenedExchange` name remains an alias for existing HSE scripts. For PBE0,
`gygi_baldereschi_zero` independently derives the zero-mode coefficient from
cell geometry, the plane-wave cutoffs and q mesh; no hybrid orbitals, fitted
constants or exchange-action labels enter this calculation. It reproduces
QE 7.2's auxiliary-function sum. On pristine LiF at 100/400 Ry and a 2x2x2
mesh, it gives 265.9096520459 bohr^2, compared with QE's 265.9096520461 bohr^2.
The full exchange actions agree to within 1.75e-12 eV.

The main experiment reuses byte-identical PBE input files from the completed
LiF HSE dataset. It generates new self-consistent PBE0 targets at the same
16 geometries, cutoffs, two q points per direction and eight bands. It uses
the same 8/2/6 split, seeds 31 and 47, 36 epochs, 288 updates per model, network
sizes, normalization scales and validation selection. All models are trained
from scratch on the new target. Three one-thread CPU workers run independent
fits; this concurrency setting is not an application-speed benchmark.

A separate pristine-cell 3x3x3 calculation checks mesh sensitivity. It does
not enter training or checkpoint selection. The two mesh settings are also
evaluated using the fixed fitted models, to expose sensitivity to changing
the integration mesh and PBE electronic inputs. This one-geometry diagnostic
is separate from the six-structure held-out comparison. A 2x2x2 mesh with
correct singularity treatment still does not establish bulk convergence.

The run is archived under `runs/lif_pbe0_exchange_v1`, including a source
snapshot, pseudopotential provenance through the source dataset, copied PBE
file hashes, input requests, native exchange actions and fitted checkpoints.
Reproduction in fresh output directories:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-qe/bin/python -m mapho.qe.exchange pbe0-dataset \
  --source runs/lif_hse_exchange_v1/dataset --output runs/lif_pbe0_dataset

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m mapho.exchange_train --dataset runs/lif_pbe0_dataset \
  --output runs/lif_pbe0_fit --epochs 36 --seeds 31 47
```

The optional training `--variants` argument selects independent architectures
when dispatching parallel fits; its default still runs all three variants.
The code suite passes 73 tests. New tests verify the Coulomb coefficient's
length-scaling law, require an explicit unscreened zero mode, and check its
occupied-projector contribution, Hermiticity and negative exchange sign.

### Completed PBE0 comparison

The matched six-structure test set gives the following mean exchange-action
RMSEs over seeds 31 and 47, in meV. The input PBE files are identical to those
used for HSE; all target actions come from newly converged PBE0 calculations.

| Predictor | All eight bands | Seed range | Six edge bands | Three empty bands |
|---|---:|---:|---:|---:|
| PBE occupied-state reference | 28.077 | — | 29.232 | 25.627 |
| MACE | 16.987 | 16.455–17.519 | 17.601 | 14.330 |
| MACE + pointwise control | **14.775** | 14.661–14.889 | **15.129** | **9.177** |
| MACE + FNO | 15.354 | 15.247–15.461 | 16.144 | 10.433 |

FNO reduces action RMSE relative to MACE alone by 9.61%; the pointwise control
is 3.77% better than FNO. Pointwise wins on all six structures in both seeds,
for both the all-band and six-edge-band metrics. The corresponding HSE values
were 9.96% and 4.01%. Removing screening therefore does not reverse the
architecture ranking in this experiment. Both combined models update the
MACE backbone and their correction branch; all six fits select epoch 36.

For withheld target k points, errors are 16.925, 14.590 and 15.223 meV for
MACE, pointwise and FNO. For the two withheld empty bands they are 10.353,
5.801 and 7.082 meV. Eight-state Ritz MAEs are 4.860, 2.687 and 2.954 meV,
versus 9.328 meV for the PBE occupied-state reference. These retain the exact
PBE0 local Hamiltonian and eigensubspace; they are not full predicted PBE0
bands. FNO's upper-valence Ritz MAE is slightly smaller than pointwise
(2.925 versus 3.272 meV), so the action ranking is not universal across
all spectral diagnostics.

The known Fock convolution already supplies the full nonlocal interaction.
These models learn a restricted scalar modulation of that operator. This
result establishes no FNO-specific advantage for that correction, and does
not test whether a more general learned nonlocal operator would benefit
from Fourier layers. The small cells, two seeds and fixed hyperparameters
also limit generalization beyond this pilot.

The completion audit passes for all 16 PBE/PBE0 pairs, six checkpoints and
3,072 evaluation rows. Maximum outer exchange residual is 9.79e-11 Ry;
native ACE differs from its construction-state exact action by at most
4.61e-14 eV per coefficient. The independent double-precision FFT check on
the first distorted training cell differs by at most 1.30e-12 eV. Peak
training-process RSS is 1,766 MiB; this is CPU memory, not GPU allocation.

### Separate mesh and CUDA check

The completed diagnostic uses an NVIDIA A100-SXM4-40GB on `debug-gpu`
(Slurm job 16240546). Repeated exchange FFT actions run on CUDA; the small
coefficient predictors remain on CPU. The earlier six training fits and
six-structure test evaluation were completed on CPU. The GPU job is not
an end-to-end timing comparison against QE.

These results use one pristine cell, excluded from training and validation,
and the fixed models trained on the 2x2x2 mesh. Values are action RMSEs in
meV, averaged over the two seeds where applicable.

| Predictor | 2x2x2 mesh | 3x3x3 mesh |
|---|---:|---:|
| PBE occupied-state reference | 27.999 | 30.953 |
| MACE | 16.676 | 17.783 |
| MACE + pointwise control | **14.730** | **15.379** |
| MACE + FNO | 15.260 | 16.057 |

Pointwise remains better than FNO for both seeds at 3x3x3. The reference
PBE0 sampled gap changes from 12.213498 to 12.187057 eV (−26.441 meV);
both sampled extrema occur at Gamma. Thus the coarse mesh has measurable
sensitivity even though the model ordering survives this check. Changing
the mesh also changes the input PBE density matrix; this is a one-cell
mesh-transfer diagnostic, not another independent training comparison or
a claim of converged bulk accuracy.

At 3x3x3, a direct double-precision full-mesh reconstruction differs from
QE's native action by 7.43e-8 eV. Inspection of QE 7.2 identifies two
implementation conventions: its auxiliary exchange list reuses the first
member of each time-reversed pair, and `vexx_k` evaluates `1.0 / nqs` in
default REAL precision before assigning it to REAL(DP). For 27 q points,
this produces a multiplicative factor 1.0000000074505806 relative to exact
uniform weights. Matching those source-level conventions in the validation
calculation reduces the error to 2.14e-12 eV across all 27 k points. No
coefficient is fitted to target actions. The model evaluation continues
to use all exported PBE orbitals and normalized weights. The direct residual
is below 0.000075 meV, negligible beside the reported model errors. CPU and
CUDA double-precision actions agree within 5.37e-15 eV at Gamma for both
meshes. Both raw and convention-matched errors are retained in the archive.

See the [PBE0 summary](../runs/lif_pbe0_exchange_v1/summary.json),
[per-state CSV](../runs/lif_pbe0_exchange_v1/evaluation/evaluation_rows.csv),
[completion audit](../runs/lif_pbe0_exchange_v1/audit.json), and
[mesh diagnostic](../runs/lif_pbe0_exchange_v1/mesh_check/summary.json).
The CUDA diagnostic can be rerun with
`sbatch runs/lif_pbe0_exchange_v1/gpu_check.sbatch` on this cluster.

## Controlled spectral-mixing comparison

The [controlled ablation protocol](spectral_ablation.md) addresses the local
capacity and short-training confounds in the initial comparisons. It compares
the same width-sixteen correction with its spectral branch removed, retained,
restricted to the cell average, or replaced by additional pointwise capacity.
The LiF PBE0 sweep uses five paired seeds and both frozen and joint MACE phases.
Its results are separate from the short-budget pilot numbers above.

## Validation and sources

Tests check linearity with complex coefficients, Hermiticity, exchange sign,
occupied-orbital unitary invariance, periodic translations, gate gradients,
FFT normalization and agreement with an explicit two-position kernel.
An independent double-precision FFT calculation must reproduce QE's hybrid
exchange actions at all eight k points to below 1e-8 eV before fitting.

- [QE 7.2 exchange and ACE implementation](https://github.com/QEF/q-e/blob/qe-7.2/PW/src/exx.f90)
- [QE 7.2 screened Coulomb convention](https://github.com/QEF/q-e/blob/qe-7.2/PW/src/exx_base.f90)
- [Adaptively compressed exchange](https://arxiv.org/abs/1601.07159)
- [Fourier neural operator](https://arxiv.org/abs/2010.08895)
- [SG15 scalar pseudopotential distribution](http://www.quantum-simulation.org/potentials/sg15_oncv/)
- [QE exchange singularity and screening options](https://www.quantum-espresso.org/Doc/INPUT_PW.html#exxdiv_treatment)
