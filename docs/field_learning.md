# MACE and Fourier residual learning of QE potentials

The runnable pilot learns the converged Hartree–XC potential of nonmagnetic,
norm-conserving PBE silicon. QE retains the kinetic operator, ionic local
potential, and nonlocal pseudopotential projectors. The trained field can be
inserted into a fresh QE calculation to obtain bands and density without SCF.

This is a small DFT learning experiment, not evidence of material transfer,
production accuracy, long-range generalization, or state-of-the-art performance.
The model does not provide consistent total energies or forces.

## What is reused

`mapho.field_model.LocalMACEField` directly constructs
`mace.modules.models.MACE` from the installed MACE package. Its two interaction
layers and symmetric many-body product blocks are trained on field labels.
The MACE energy readouts are frozen and unused. This is a new, small model
trained from scratch; it does not load a pretrained MACE force-field checkpoint.

The new e3nn field head maps concatenated MACE node features to coefficients
of compact radial functions and real spherical harmonics at spatial probes.
It sums every periodic atom image within the probe cutoff. A scalar head also
learns a constant offset. No target-derived mean is supplied at inference.
The decoder queries a periodic spatial index in bounded probe/neighbor chunks.
It recomputes local basis functions during backpropagation instead of storing
a dense grid-point-by-atom basis. See [memory scaling](memory.md).

The compact FNO is implemented in this package, following the physical
reciprocal-space filtering approach of the adjacent `mace_fno` project.
It does not import that project's energy model or interpret its energy field
as an electronic potential. No files in that project were modified.

## Model and training

```text
V_Hxc(predicted) = V_Hxc(atomic reference)
                  + delta_V_local_MACE
                  + delta_V_FNO
```

The target is the full converged-minus-reference Hxc field in eV, including
its mean. The native QE gauge, functional, pseudopotential hash, reference
definition, and solver settings must match the training contract.

The pilot uses:

- Two MACE interactions; eight channels per irrep, angular momenta 0–2,
  correlation order three, and a 4.5 angstrom atomic cutoff.
- A 3 angstrom probe cutoff and 12 Gaussian radial functions, with a smooth
  cutoff and powers of radius to regularize angular functions at nuclei.
- A 16×16×16 training mesh. Labels are periodically Fourier-resampled from
  QE's native mesh, with explicit treatment of even-grid Nyquist frequencies.
- A two-layer, eight-channel FNO. Its three inputs are the local prediction,
  atomic-reference density, and atomic-reference Hxc. Real channel-mixing
  filters depend on `|2 pi A^-1 n|²`, with a smooth 6 inverse-angstrom
  frequency cutoff. Nyquist planes are excluded from the spectral branch to
  preserve conjugacy in skew cells. A pointwise branch supplies local mixing.
- Sequential fitting: select the local checkpoint on validation loss, freeze
  it, then train the FNO on its remaining error. Normalization uses training
  structures only. Select the FNO checkpoint on validation loss. An optional
  fine-tuning stage subsequently updates both models together.

At inference, the local decoder is evaluated on the native QE grid. The FNO
is evaluated on the training mesh using the local prediction on that same
mesh, then Fourier-interpolated to the native grid. This is interpolation,
not a demonstration of resolution-independent accuracy.

All scalar meshes use fractional lattice axes `(i,j,k)`, unlike the
`(z,x,y)` storage used in parts of `mace_fno`. Cells store direct vectors
as rows. A rigid Cartesian rotation leaves the physical spectral metric
unchanged. Grid translations commute with the FNO; the local head is also
tested under rotations, atom permutations, and periodic translations.

## Environments

Use the existing learning environment with PyTorch, e3nn, MACE, ASE, NumPy
and SciPy. The tested combination is MACE 0.3.16, PyTorch 2.7.1 and e3nn 0.5.9.
This workspace reuses MACE from the user's installation via a project virtual
environment with system site packages. The user's installation was not changed.
The separately pinned QE runtime is described in [qe.md](qe.md).

MACE 0.3.16's package metadata pins e3nn 0.4.4. Therefore an ordinary combined
dependency resolution with this project's e3nn requirement does not reproduce
the tested environment. The field extra supplies SciPy; it does not install
or upgrade MACE. Reuse a working MACE environment and verify compatibility
with the field tests. Do not upgrade the user's force-field environment.

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

The current field implementation and commands run on CPU in float32. QE runs
in float64 in a separate process. A CPU-only optional-acceleration warning from
the installed MACE package does not prevent execution.

## Generate, train, and evaluate

Run from the `mapho` directory. All output directories must be new.

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-qe/bin/python -m mapho.qe.field_dataset \
  --pseudo runs/qe_si_roundtrip/Si_ONCV_PBE-1.2.upf \
  --output runs/si_fields_new --count 32 --seed 29

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train train \
  --dataset runs/si_fields_new --output runs/si_model_new \
  --local-epochs 160 --fno-epochs 100 --seed 31

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train evaluate \
  --dataset runs/si_fields_new --checkpoint runs/si_model_new/model.pt \
  --qe-python .venv-qe/bin/python --output runs/si_model_new/evaluation
```

The generator uses two-atom diamond Si, independent isotropic strains within
±1.5%, symmetric strain perturbations within ±0.6%, and second-atom Cartesian
displacements within ±0.025 angstrom per component. It stores a deterministic
20/6/6 structure split, full QE labels and provenance, and a completion marker.
It rejects nonpositive gaps on the sampled k mesh. A wider initial strain
range produced band overlap and was abandoned before training; fixed
occupations are unsuitable there.

DFT settings are 40/160 Ry cutoffs, a 2×2×2 k mesh, eight bands, and the
checksum-pinned SG15 Si ONCV PBE 1.2 pseudopotential. This coarse k mesh is a
pipeline benchmark, not a converged Si band-gap calculation.

Training writes `model.pt`, `history.json`, `runtime.json` and `split.json`.
Structures are loaded one at a time. After freezing the local model, compact
per-structure FNO inputs and residual targets are written to `fno_cache/`
and streamed from disk. Atomic graphs and probe bases are not cached across
the dataset or retained for FNO training.
Evaluation uses the stored test split, writes each full potential and QE
solution, and reports equal-weight structure averages in `metrics.json`.
Absolute band errors preserve the native gauge; separately reported aligned
errors subtract one fitted spectral shift per structure. A sampled gap is the
lowest empty minus highest occupied eigenvalue across the eight sampled k
points, not a dense band-path result.

The atomic baseline means diagonalizing the Hamiltonian constructed with
the atomic-reference Hxc; its reported density is that diagonalization's
density, not the atomic-superposition input density.

## Joint fine-tuning and its control

The `finetune` command starts from an existing field checkpoint. With
`--mode joint`, it recomputes the local prediction at every training step and
minimizes the error of the combined residual:

```text
local = MACE_field(structure)
correction = FNO(stack(local, reference_density, reference_Hxc), cell)
loss = mean((local + correction - target_residual)^2) / output_scale^2
```

Gradients reach MACE through both the additive local prediction and its role
as an FNO input. Neither path is detached. The MACE backbone and field heads
are trainable; its unused energy readouts remain frozen. Probe contraction
still recomputes bounded basis batches in backward. Joint training streams
one structure at a time and does not create a frozen local prediction cache.

`--mode fno` is a control: keep MACE fixed and continue optimizing only FNO.
This mode caches fixed inputs and residuals on disk, as in staged training.
Both modes retain the checkpoint's input/output normalization, start fresh
Adam optimizers, and use cosine decay to 3% of each group's initial learning
rate. This is continuation from weights, not restoration of optimizer state.
The combined validation potential MSE selects checkpoints, including epoch
zero, so an unsuccessful continuation can retain its starting weights.
The original manifest hash and physical contract must match.

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train finetune \
  --dataset runs/si_fields_v2 --checkpoint runs/si_mace_fno_v1/model.pt \
  --output runs/joint_new --mode joint --epochs 50 \
  --local-lr 0.0001 --fno-lr 0.0003 --seed 47

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train finetune \
  --dataset runs/si_fields_v2 --checkpoint runs/si_mace_fno_v1/model.pt \
  --output runs/fno_continuation_new --mode fno --epochs 50 \
  --fno-lr 0.0003 --seed 47

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train evaluate \
  --dataset runs/si_fields_v2 --checkpoint runs/joint_new/model.pt \
  --models local_fno --qe-python .venv-qe/bin/python \
  --output runs/joint_new/evaluation
```

Matching epochs and seeds matches sample order and optimizer update counts;
it does not match wall time or floating-point operation counts. Metadata
records parent checkpoint hashes, learning rates, selected epochs, and
parameter changes separately for the backbone, heads, and FNO. The starting
checkpoint is never overwritten. Once fine-tuned jointly, the local branch
alone need not retain its previous accuracy: validation selects the sum.

## Completed pilot

The workspace run uses `runs/si_fields_v2` and
`runs/si_mace_fno_v1/model.pt`. Both checkpoints were selected by validation
loss, before the six test structures were evaluated. Mean results on the
native 30×30×30 QE grids are:

| Model | Potential RMSE (eV) | Absolute band MAE (meV) | Sampled gap MAE (meV) | Relative density L1 (%) |
|---|---:|---:|---:|---:|
| Atomic reference | 0.7136 | 611.7 | 385.2 | 3.384 |
| Local MACE | 0.2577 | 27.9 | 71.8 | 0.559 |
| Local MACE + FNO | 0.1542 | 7.0 | 8.9 | 0.129 |

The band statistic averages eight bands and eight k points, then weights
each structure equally. No spectral shift is removed in the table. Hybrid
band MAEs range from 5.74 to 9.01 meV across the six structures. Potential
RMSE retains the complete native-grid error, including modes beyond the
training mesh. Density error is `integral |rho_pred-rho_ref| / Ne`.

Full per-structure results and predictions are in
`runs/si_mace_fno_v1/evaluation/metrics.json`. There are 8,469 local-model
parameters (including the unused frozen energy readouts) and 2,425 FNO
parameters. The validation-selected epochs were 160 and 100 respectively;
validation potential RMSEs on the 16³ mesh were 0.2481 and 0.1381 eV.

The suite passes 64 tests, including joint gradient flow, frozen-control
checkpoint checks, and explicit band-path requests. A separate SCF-free inference check is stored in
`runs/si_atomic_inference`: its initialization log has zero SCF iterations,
and its predicted-potential solve agrees with the corresponding evaluation
solve within 2.4e-8 eV in eigenvalues. Tiny input differences arise from
extended-XYZ coordinate rounding. The atomic reference format contains no
converged labels; tests also alter all converged labels and verify predictions
are unchanged.

## Completed fine-tuning comparison

`runs/si_joint_finetune_v1` compares the staged checkpoint with FNO-only
continuation and joint fine-tuning. The protocol was written before either
training run. Each continuation uses 50 epochs (1,000 optimizer updates),
seed 47 and FNO learning rate 0.0003. Joint training uses MACE learning rate
0.0001. Both select epoch 50 using validation potential error; their initial
validation RMSE is identical at 0.138054 eV. It decreases to 0.133408 eV in
the frozen control and 0.117219 eV with joint training on the 16³ mesh.

Fresh QE solves for all three checkpoints on the same six test structures give:

| Model | Native potential RMSE (eV) | Absolute band MAE (meV) | Sampled gap MAE (meV) | Relative density L1 (%) |
|---|---:|---:|---:|---:|
| Original staged model | 0.15418 | 7.042 | 8.882 | 0.12880 |
| FNO-only continuation | 0.15035 | 6.935 | 9.443 | 0.12916 |
| Joint fine-tuning | 0.13717 | 6.528 | 9.617 | 0.12421 |

Joint training improves average potential RMSE, band MAE and density error,
but slightly worsens the sampled gap error. Potential-error minimization
does not guarantee improvement in every electronic observable. These are
results from one continuation seed on a narrow silicon distribution; they
do not establish general superiority or a long-range learning mechanism.

The matched budget is optimizer updates, not computational cost: the FNO-only
process took about 19 seconds and joint training about 102 seconds on this
CPU run, including startup. Frozen-local caching makes the control cheaper.
Separate QE evaluation took about 28–30 seconds per checkpoint. These are
observed run times, not controlled performance benchmarks.

`verification.json` confirms that 27 backbone parameter tensors, both field
heads and all 16 FNO parameter tensors changed during joint fine-tuning.
All local parameters and buffers remain exactly unchanged in the frozen
control. Normalization and unused energy heads remain unchanged in both.
The parent checkpoint SHA-256 remains
`775832e5b9e9fbceef0647fad136ffd525d954da83d70d1cc80ba5dacde2066a`.

The run directory includes `protocol.json`, `run_experiment.py`, histories,
new checkpoints under `fno/` and `joint/`, all potential predictions and QE
solutions under `evaluation_*`, and aggregate `comparison.json`. The original
checkpoint is retained; the joint checkpoint is an optional alternative with
the tradeoffs shown above.

## Inference without SCF labels

```bash
env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe atomic \
  --structure structure.extxyz --pseudo Si=/path/to/Si.upf \
  --output runs/new_atomic

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train predict \
  --reference runs/new_atomic/atomic.npz --checkpoint runs/si_model_new/model.pt \
  --output runs/new_atomic/predicted_hxc.npy

env -u PYTHONPATH .venv-qe/bin/python -m mapho.qe solve \
  --reference runs/new_atomic/atomic.npz --hxc runs/new_atomic/predicted_hxc.npy \
  --output runs/new_solve
```

`atomic.npz` contains geometry, initial reference fields, k points and
operator metadata. It contains no converged density, converged potential,
eigenvalues, or wavefunctions. The export calls QE initialization without
`driver.scf()`. The final solve starts from scratch with the supplied full
Hxc and never updates it self-consistently. The exact pseudopotential is still
required at inference.

## Limits of this experiment

Two-atom cells and one narrow silicon distribution cannot establish the
benefit of global interactions. The FNO adds both spectral and pointwise
capacity; an improvement over the local model alone does not isolate a
long-range mechanism. Larger cells, controlled pointwise-only ablations,
separated defects/interfaces, and split-by-family tests are needed.

This prototype also needs mesh/cutoff convergence, broader k-point evaluation,
and SCF residual checks. Probe evaluation now has bounded neighbor batches;
the FNO still requires full scalar feature meshes and FFT buffers. The local
decoder supports first derivatives of model parameters, but geometry
derivatives and higher-order autograd through its fixed probe index are unavailable.
There is no claim of covariance under every lattice-basis change on a finite
mesh. Metallic smearing, spin/SOC, PAW/ultrasoft, slab electrostatics, and
VASP/ABINIT interfaces remain separate work.
