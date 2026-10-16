# Tutorial: PBE Hamiltonian learning and Si bands

This tutorial trains MAPHO on distorted 16-atom Si cells and applies the
resulting model to primitive Si. We prepare the reference operators, train
one MAPHO-FNO model from YAML, and compare the predicted bands with PBE on
the X–Gamma–L path. All calculations use nonmagnetic PBE without SOC.

The example illustrates the complete calculation sequence. It does not
provide a pretrained model or a claimed accuracy for a new training run.

## Contents

- [Environment and paths](#environment-and-paths)
- [1. Data preparation](#1-data-preparation)
- [2. Training](#2-training)
- [3. Inference and band structure](#3-inference-and-band-structure)
- [Interpretation of the comparison](#interpretation-of-the-comparison)
- [Other materials and common failures](#other-materials-and-common-failures)

## Environment and paths

Install the dependencies described in [usage.md](usage.md#installation).
The calculation requires a private MAPHO-enabled VASP executable and a
local PBE Si PAW dataset. The executable must support `MAPHO_EXPORT` and
`MAPHO_IMPORT`. The required private patches are
[not included in the public repository](src/mapho/vasp/README.md).

Run the following setup from the root of the repository. Replace the three
placeholder paths. Use scratch storage for `MAPHO_WORK` and a new directory
for this example.

```bash
MAPHO_ROOT="$PWD"
MAPHO_PYTHON="$MAPHO_ROOT/.venv/bin/python"
MAPHO_WORK=/absolute/path/on/scratch/mapho_tutorial
MAPHO_POTCAR=/absolute/path/to/Si/POTCAR
MAPHO_VASP=/absolute/path/to/mapho-enabled/vasp_std
MAPHO_DATASET="$MAPHO_WORK/si_data"
MAPHO_ATOMIC="$MAPHO_DATASET/atomic"
mkdir -p "$MAPHO_WORK"

mapho_cmd() {
  env -u PYTHONPATH "$MAPHO_PYTHON" -m mapho.commands.vasp "$@"
}
```

The helper selects the project environment without inheriting an unrelated
`PYTHONPATH`. If the environment is elsewhere, change `MAPHO_PYTHON`.
Keep these variables in each job script, or define them again in each shell.

Input preparation can run on a login node. Execute the VASP calculations
and model training inside an A100 allocation with the matching compiler and
MPI runtime. The commands below do not request a Slurm allocation. The data
drivers distribute independent structures across the allocated GPUs, using
one MPI rank per VASP process. Training uses one GPU.

## 1. Data preparation

### Target and reference operators

Each geometry has two operator files:

| Operator | Calculation | Role |
|---|---|---|
| Target | Converged PBE SCF | Training labels |
| Atomic reference | Fixed superposed atomic densities, `ICHARG=12` | Input and residual reference |

The operators contain the smooth potential, full PAW D matrices, overlap
matrices Q, projector-channel metadata, geometry, and provenance. The model
learns the differences in the potential and D matrices. The kinetic energy,
projectors, and overlap remain part of the native eigensolver.

### Structures and input files

```bash
mapho_cmd data silicon prepare "$MAPHO_DATASET" \
  --potcar "$MAPHO_POTCAR" --executable "$MAPHO_VASP" --count 24
```

This command constructs 2×2×2 primitive supercells with 16 Si atoms from
diamond Si at a conventional lattice constant of 5.43 Å. It applies small
strains and atomic displacements, then assigns 16 structures to training,
four to validation, and four to testing. Calibration structures are separate
from these splits.

Pymatgen generates Gamma-centered meshes using reciprocal-volume densities
of 200, 300, and 500. These densities are inputs to the mesh generator, not
numbers of k points. The actual mesh of every structure is recorded in the
manifest and `KPOINTS`. The inputs use a fixed cutoff of 600 eV and 48 bands.
Cutoff convergence remains a separate check.

Preparation writes `POSCAR`, `INCAR`, `KPOINTS`, `POTCAR`, and `MAPHO_EXPORT`
for each calculation. It also records the executable and source hashes.
The output directory must not already exist.

### PBE calculations and data audit

Inside the GPU allocation, run the calculation driver using the source
snapshot created during preparation:

```bash
PYTHONPATH="$MAPHO_DATASET/source_snapshot" "$MAPHO_PYTHON" \
  -m mapho.commands.vasp data silicon execute "$MAPHO_DATASET"
mapho_cmd data material audit "$MAPHO_DATASET"
```

The shared audit currently uses the command name `data material audit` for
Si as well. The execution driver first calculates the calibration structures.
Production calculations start only if the sampling checks pass. The checks
compare adjacent meshes for total energy, common-k eigenvalues, the potential,
and PAW matrices. They require 1 meV/atom in energy, 1 meV for the selected
eigenvalues, and 5 meV RMSE for each field or matrix comparison.

The driver exports `POT` and `MAPHO_PAW.dat` through the private interface,
then packages the data as `operator.h5`. The audit checks completed native
runs, file identities, finite arrays, and the sampling result. Successful
execution produces `completion.json` and `data_audit.json`.

The Si driver is intended for a short allocation and has a total dispatch
budget of about 115 minutes. It is not a general restart manager. Preserve
failed or incomplete outputs and inspect the logs before restarting work.

### Atomic references

```bash
mapho_cmd data atomic-reference prepare "$MAPHO_DATASET" "$MAPHO_ATOMIC"
mapho_cmd data atomic-reference execute "$MAPHO_ATOMIC"
```

The preparation command copies the geometry and PAW choices, and matches
the FFT grid of each target. The atomic calculation starts without a target
`CHGCAR` or `WAVECAR`. `ICHARG=12` fixes the superposed atomic densities and
PAW occupancies. These are separate calculations, not copies of the
converged target potential. The execution command also requires a GPU
allocation and writes `atomic/completion.json` when all cases finish.

The resulting training inputs have the following layout:

```text
si_data/
  manifest.json
  convergence.json
  completion.json
  data_audit.json
  samples/sample_000/
    POSCAR
    operator.h5
    ... native inputs and outputs ...
  atomic/
    manifest.json
    completion.json
    sample_000/operator.h5
    ... matching references for all samples ...
```

Keep the native calculation directories and private PAW files at their
recorded locations. The operators refer to these files for later replay.

## 2. Training

### YAML recipe

Create the recipe next to the dataset. Relative paths are resolved from the
directory containing this YAML file.

```bash
cat > "$MAPHO_WORK/train.yaml" <<'YAML'
version: 1
dataset: ./si_data
atomic_reference: ./si_data/atomic
output: ./si_fno
target: pbe
mode: full
correction_activation: gelu
model:
  atom_cutoff: 4.5
  probe_cutoff: 3.0
  num_interactions: 2
  channels: 8
  lmax: 2
  correlation: 3
  radial_count: 12
  grid_shape: [32, 32, 32]
  fno_width: 8
  fno_layers: 2
  q_cutoff: 6.0
sampling:
  point_count: 8192
training:
  epochs: 300
  seed: 31
  device: cuda
  backbone_lr: 3.0e-4
  head_lr: 1.0e-3
YAML
```

`num_interactions` sets the number of MACE interaction layers. `lmax` sets
the angular resolution. The maximum reach of the local interaction and
probe path is `2 × 4.5 + 3 = 12 Å` for these settings. The scalar offset and
the Fourier correction also involve whole-cell information.

`fno_width: 8` specifies eight hidden feature channels. The spectral branch
selects FFT wavevectors with `|q| < 6 Å⁻¹`, excluding Nyquist planes. The
number of selected modes depends on the cell and the grid. It is independent
of the feature width. The [training guide](docs/training.md) explains these
settings and the optimizer defaults.

### Fitting and evaluation

Inspect the recipe, then train inside the GPU allocation:

```bash
mapho_cmd train run --config "$MAPHO_WORK/train.yaml" --dry-run
mapho_cmd train run --config "$MAPHO_WORK/train.yaml"
```

The second command checks the data, constructs the cache, fits the model,
and evaluates the held-out potentials and PAW matrices. The MACE backbone,
field decoder, PAW heads, and Fourier correction are optimized jointly. The
loss combines field and PAW-entry mean-square errors, normalized by scales
obtained from the training split. The validation objective selects the
checkpoint. The test split does not select the model.

The principal outputs are:

| File under `si_fno/` | Content |
|---|---|
| `config.input.yaml`, `config.resolved.json` | Supplied recipe and effective settings |
| `manifest.json` | Architecture, input identities, normalization, and splits |
| `full/model.pt` | Selected checkpoint |
| `full/history.json` | Training and validation history |
| `full/training.json` | Optimization details, parameter changes, and timing |
| `full/test_fields.json` | Held-out potential and PAW errors |

To train MAPHO-MACE on the same data with a separate output directory:

```bash
mapho_cmd train run --config "$MAPHO_WORK/train.yaml" \
  --mode local --output "$MAPHO_WORK/si_mace"
```

This optional comparison is not required for the band example. A new mode
or depth requires a new training run, not an edit to a saved checkpoint.

## 3. Inference and band structure

### Primitive Si example

The [band example](examples/vasp/si_bands.py) accepts the single checkpoint
created above. Run it inside an A100 allocation with the same VASP runtime:

```bash
env -u PYTHONPATH "$MAPHO_PYTHON" "$MAPHO_ROOT/examples/vasp/si_bands.py" \
  --training "$MAPHO_WORK/si_fno" \
  --mode full --output "$MAPHO_WORK/si_bands"
```

The example uses two-atom diamond Si at 5.43 Å. It recovers the private
executable and PAW dataset from the training provenance and checks their
hashes. The default MPI launcher follows the existing single-rank OpenMPI
workflow. `--launcher 'srun -n 1'` can replace it on a system configured for
that launcher. `--device cpu` changes the device used by the ML model, not
the executable used by VASP.

The calculation sequence is:

1. Construct an atomic reference for the primitive geometry with `ICHARG=12`.
2. Evaluate MAPHO and save the predicted potential and PAW matrices.
3. Calculate independent PBE references on 14×14×14 and 16×16×16 meshes.
   Compare their energies, common-k eigenvalues, potentials, and PAW matrices.
4. Replay the denser PBE operator on its original mesh. Require agreement
   with native eigenvalues within 0.1 meV.
5. Diagonalize the PBE and predicted operators along the same 81-point
   X–Gamma–L path, using eight bands.

The prediction is saved before the independent PBE calculations. Their
electronic labels are used only for assessment. Both path calculations use
`ICHARG=14` with the supplied operator. The PBE comparison is therefore a
frozen exported PBE operator, not a separate `ICHARG=11` reconstruction
from `CHGCAR`. The native replay check assesses the export/import error.

The script stops if either the sampling check or replay check fails. The
default primitive meshes must pass these checks in the actual calculation.
`--meshes 16 18` requests another increasing pair in a new output directory.

### Inference API

The essential prediction step is:

```python
from mapho.vasp.train import load_model, predict
from mapho.vasp.paw import prepare_replay, sha

model = load_model(training_directory, "full", "cuda")
potential, d = predict(
    model,
    {"baseline": atomic_h5, "baseline_sha256": sha(atomic_h5)},
    "cuda",
)
prepare_replay(
    atomic_h5, band_directory, potential=potential, d=d, kpoints=kpoints
)
```

Here, `atomic_h5` is the operator of the new atomic reference. The variables
denote existing paths, and `kpoints` contains fractional reciprocal
coordinates. `prepare_replay` writes the supplied potential and matrices
for VASP. The example then calls the native solver to obtain eigenvalues.
VASP reads these arrays, not the PyTorch checkpoint.

### Output

| File under `si_bands/` | Content |
|---|---|
| `prediction.npz` | Predicted smooth potential and PAW D matrices |
| `convergence.json` | Primitive-cell sampling comparison |
| `replay_check.json` | Native operator replay comparison |
| `bands.pdf`, `bands.png` | PBE and MAPHO bands, with frontier errors below |
| `band_source_data.csv`, `bands.npz` | Every calculated level on the path |
| `metrics.json` | Frontier errors and path gaps |
| `completion.json` | Written after all stages pass |

The figure uses gray short dashes for PBE and a thin solid line for MAPHO.
The lower panel shows signed errors of the top occupied and bottom empty
bands. Both spectra share the VBM of the PBE reference as the plotting zero.
The numerical errors are calculated from the unshifted eigenvalues.

## Interpretation of the comparison

The frontier MAE covers two occupied and two empty bands at every path
point. The lower plot shows only the outermost occupied and empty bands.
These are distinct summaries of the same spectra. A path gap is the gap
sampled on X–Gamma–L, not a search over the full Brillouin zone.

This calculation assesses transfer from distorted Si16 cells to pristine
primitive Si without retraining. It does not establish transfer to another
element, PAW dataset, charge state, or functional. For semilocal PBE, the
potential and PAW matrices have no explicit k dependence, so one prediction
can be diagonalized at all path points. Native diagonalization and the
atomic-reference calculation remain part of the cost of inference.

## Other materials and common failures

The current material generators provide LiF and bilayer MoS2 through
`mapho-vasp data material prepare`. These are prescribed material protocols,
not a generic importer for arbitrary POSCAR collections. The primitive Si
band example is restricted to scalar PBE Si checkpoints. PBE0 needs the
separate [reference-exchange workflow](docs/vasp_pbe0.md).

| Failure | Action |
|---|---|
| Sampling check fails | Inspect `convergence.json` and prepare a new calculation with denser sampling. Do not change a failed flag manually. |
| Export/import marker is absent | Check that the private executable supports the MAPHO protocol. |
| Executable, PAW, or source hash differs | Use the recorded files and source snapshot, or prepare a new study. |
| Output directory exists | Select a new path. The tutorial commands do not resume an interrupted calculation. |
| Atomic reference finds `CHGCAR` or `WAVECAR` | Use a fresh atomic-reference calculation without electronic restart files. |

Input, operator, and checkpoint conventions are described in the
[training guide](docs/training.md). The current imported-operator workflow
provides electronic bands, not validated total energies, forces, or phonons.
