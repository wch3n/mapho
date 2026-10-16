# Memory scaling of the field model

The production field path streams local probe interactions and dataset
samples, and evaluates spectral filters in bounded batches. Existing trained
checkpoints load without conversion or retraining. New chunk-size settings
control temporary storage and do not change learned parameter shapes.

## Local decoder

`PeriodicProbeGrid` in `mapho.probes` indexes relevant periodic atom images
with a Cartesian cKDTree. Reciprocal plane heights determine conservative
image bounds for arbitrary nonsingular, right-handed cells. Each batch queries
only atoms within the probe cutoff, including multiple images of an atom.
Probe coordinates are generated from flat mesh indices as needed.

MACE is evaluated once per structure during inference. Its atomic expansion
coefficients are reused for the native and coarse meshes. Each batch computes
radial functions and spherical harmonics, contracts them with neighboring
atomic coefficients, and adds the scalar contributions to the output.

The custom autograd contraction saves no edge basis tensors. Backward
recomputes those tensors in batches and accumulates coefficient gradients,
which propagate through the field head and MACE. This exchanges computation
for memory while preserving the original first-order parameter gradients.
The probe geometry is fixed: geometry derivatives and higher-order autograd
through this contraction are unsupported.

The removed dense basis required

```text
4 * Ngrid * Natoms * Nradial * (lmax+1)^2 bytes  [float32]
```

The largest single edge-basis batch now requires

```text
4 * edge_chunk_size * Nradial * (lmax+1)^2 bytes
```

At defaults (2,048 edges, 12 radial functions, lmax=2), this is 0.844 MiB.
Coefficient gathers, products, neighbor lists and index arrays require
additional temporary memory. The periodic atom index, atomic features and
full scalar output also remain. At fixed density/cutoff and reasonable cell
geometry, storage no longer includes a grid-by-atom factor.

`probe_basis` remains as a small dense diagnostic reference, with a 256 MiB
allocation limit. Production training and prediction never call it.

## Training data

`load_dataset` validates structures sequentially and returns path-backed
`FieldFiles` sequences. It retains geometry hashes for duplicate checking,
but no loaded field samples. `PreparedFields` constructs only the current
structure's graph, probe index, reference channels and target.

After selecting and freezing MACE, `cache_residuals` writes one compact NPZ
per training/validation structure under `fno_cache/`. These contain only:

- The local prediction, initial density and initial Hxc input channels.
- The residual target remaining after the local prediction.
- The cell.

`ResidualFiles` reads these one structure at a time. Training normalization
is accumulated from the training split; validation statistics are not used.
The local-stage prepared datasets are released before FNO optimization.
The extra disk I/O and probe recomputation can increase training time.

Optional joint fine-tuning regenerates local predictions at each step and
retains that structure's MACE and FNO computation graphs until backward.
It streams structures and uses the same bounded probe/spectral operations,
but cannot use frozen local prediction caches. The staged-training memory
measurements below do not describe the joint training peak.

## Spectral operator

Each layer selects only modes inside its physical spherical frequency
cutoff, with the original Nyquist-plane exclusions. The radial network
evaluates at most `spectral_chunk_size` modes per call. Its channel-mixing
matrices stay real; real and imaginary Fourier components are mixed separately.
The previous full-grid real and complex mixing tensors are eliminated.

During gradient-enabled evaluation, non-reentrant checkpointing recomputes
these mixing matrices in backward. Retained input/output coefficients scale
with channels per mode, rather than channels squared per mode. Tests compare
input, cell and parameter gradients with the original dense expression.

The full FFT grids and neural feature fields remain, as do the retained-mode
coefficients and reciprocal geometry arrays. This optimization does not make
global FFT processing independent of mesh size.

## Controls

All field `train`, `predict` and `evaluate` commands accept:

```text
--probe-chunk-size 512
--probe-edge-chunk-size 2048
--spectral-chunk-size 256
```

Smaller values reduce intermediate storage at the cost of additional calls.
For example, using the existing checkpoint:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mapho.field_train predict \
  --reference runs/si_atomic_inference/atomic.npz \
  --checkpoint runs/si_mace_fno_v1/model.pt \
  --output runs/predicted_chunked.npy \
  --probe-chunk-size 128 --probe-edge-chunk-size 512 \
  --spectral-chunk-size 17
```

## Measurements in this workspace

Measurements use separate CPU processes, float32 and one compute thread.
Peak RSS includes Python, PyTorch, e3nn and other imported libraries; QE is
excluded. Linux `/proc/self/status` VmHWM and sampled RSS were recorded,
because `ru_maxrss` in this execution environment retains a pre-exec
high-water mark. These measurements describe memory, not general performance
or electronic-structure accuracy on larger cells.

| Workload | Previous peak RSS | New peak RSS |
|---|---:|---:|
| Pilot training: one epoch in each stage, all 32 structures indexed | 0.80 GiB | 0.64 GiB |
| One spectral layer: 32³ grid, 32 channels, forward and backward | 1.36 GiB | 0.60 GiB |
| Local decoder: 128 atoms, 120³ grid | Not run; dense basis alone estimated at 89 GiB | 0.72 GiB |

The 128-atom test repeats a two-atom Si cell 4×4×4 and evaluates the local
decoder on 1,728,000 probes. It is a memory/scaling test, not new DFT validation
or a test of FNO transfer to larger meshes. Its local potential agrees with
the tiled original prediction within 2.13e-5 eV (float32 summation differences).

The six original held-out structures were also reevaluated using the original
checkpoint. The largest hybrid-potential difference is 7.91e-6 eV. A fresh
QE solve on the first held-out structure changes eigenvalues by at most
4.92e-8 eV and density by 7.58e-9 electrons/angstrom³ relative to the original
prediction's solve. No production checkpoint was retrained for this memory check.

Artifacts are under `runs/memory_optimization/`: the pre-change source copies,
`benchmark.py`, per-process JSON reports, predictions, the QE check, and
separate short training runs. The memory checks include dense
versus streamed values/gradients, periodic images and empty neighborhoods,
bounded spectral batches, sample lifetime checks and streamed training.
