# Training from YAML

The [Si tutorial](../tutorial.md) connects data preparation, YAML training,
and band inference. See [usage](../usage.md) for installation and commands.

The VASP PAW trainer accepts a YAML recipe in addition to the existing
`train model` and `train pbe0-heads` commands. It supports the scalar PBE model
and the conditional PBE0 local/PAW model. The QE workflows and the PAW repair
and joint fine-tuning workflows keep their existing interfaces.

Start with [PBE](../examples/training/pbe.yaml) or
[Si PBE0](../examples/training/si_pbe0.yaml). Set the data and output paths,
then inspect the resolved settings:

```sh
mapho-vasp train run --config examples/training/pbe.yaml --dry-run
mapho-vasp train run --config examples/training/pbe.yaml
```

In the local workspace, use `env -u PYTHONPATH .venv/bin/mapho-vasp` in place
of `mapho-vasp` to select the project environment. Run GPU training inside an
allocated A100 Slurm job. The command does not request an allocation itself.

The YAML runner prepares the cache, trains one model, and evaluates its
potential and PAW matrices on the held-out test set. It does not run VASP
or generate reference data. PBE0 eigenvalue evaluation remains a separate
`mapho-vasp train pbe0-heads evaluate-bands TRAINING full` command, using
reference ACE exchange. This is not structure-only hybrid inference.

## Configuration and overrides

```yaml
version: 1
dataset: ./data
atomic_reference: ./atomic
output: ./training
target: pbe
mode: full
correction_activation: gelu
model:
  channels: 8
  num_interactions: 2
  grid_shape: [32, 32, 32]
  fno_width: 8
  fno_layers: 2
training:
  epochs: 300
  seed: 31
  device: cuda
  backbone_lr: 3.0e-4
  head_lr: 1.0e-3
```

`full` selects MAPHO-FNO, and `local` selects MAPHO-MACE. Both learn PAW
corrections. The existing pointwise control is available as `none` for PBE.
`correction_activation` controls the grid correction; it does not replace
the internal activations of MACE. The YAML default is GELU. The existing
unconfigured PBE preparation command retains its original SiLU default.

Precedence is CLI overrides, then YAML values, then defaults. Available CLI
overrides are `--output`, `--mode`, `--epochs`, `--device`, `--seed`,
`--backbone-lr`, and `--head-lr`:

```sh
mapho-vasp train run --config train.yaml --epochs 600 --seed 47 \
  --mode local --output runs/mace_seed47
```

YAML paths are relative to the YAML file; `--output` is relative to the
working directory. `~` is expanded. There is no environment-variable
interpolation or executable YAML syntax. Unknown fields, duplicate keys,
invalid values, and unsupported targets fail before run creation.

`model` accepts the field settings in the full PBE example, plus `grid_size`
(default 32), `probe_chunk_size` (8192), `probe_edge_chunk_size` (32768), and
`spectral_chunk_size` (256). A supplied `grid_shape` takes precedence over
`grid_size`. Atomic species and PAW channels are inferred from the data.
`model.num_interactions` sets the number of MACE message-passing layers
(positive integer, default 2). It is saved in the manifest and checkpoint
and used to reconstruct the backbone at inference. Checkpoints made before
this setting was introduced retain two layers. Changing the depth requires
a new training run; it does not change the depth of an existing checkpoint.

The local message-passing and probe path has a maximum reach of
`num_interactions * atom_cutoff + probe_cutoff`. With the default cutoffs,
two layers give 12 angstroms and three give 16.5 angstroms. This describes
the local path, not the whole model: the averaged scalar offset and the
FNO involve whole-cell information. `lmax` controls angular resolution,
while `fno_layers` independently controls the depth of the Fourier correction.
With one interaction, the current MACE backbone exposes only scalar
features; at least two are needed to retain nonscalar features for the
angular decoder and PAW heads.

### Dataset and atomic-reference layout

Both paths refer to prepared MAPHO datasets, not directories containing only
raw VASP inputs and outputs. A typical PBE layout is:

```text
dataset/
  manifest.json
  data_audit.json
  samples/
    sample_000/
      POSCAR
      operator.h5
    ...
atomic_reference/
  manifest.json
  completion.json
  sample_000/
    operator.h5
  ...
```

The `splits` mapping in `dataset/manifest.json` supplies nonempty `train`,
`validation`, and `test` lists of operator paths relative to `dataset`, for
example `samples/sample_000/operator.h5`. The intermediate `samples/`
directory is optional. Sample-directory names must be unique and end in an
integer, such as `sample_000`, because that index determines the sampling
seed. Each target operator needs a neighboring `POSCAR` and a matching
`atomic_reference/<sample-directory-name>/operator.h5`.

The HDF5 operators contain the geometry, smooth potential, PAW D and Q
matrices, projector channels, and export metadata. The reference is a VASP
`ICHARG=12` calculation with fixed superposed atomic densities for the same
geometry and PAW datasets. Training learns the difference from this reference.
The dataset audit and reference completion files must report success, and
the reference manifest must match the hash of the dataset manifest. These
files are produced by the data preparation and audit workflows; they should
not be written manually to bypass the checks. If `atomic_reference` is
omitted from YAML, it defaults to `dataset/atomic`.

For conditional PBE0, split entries point to `operator_components.h5` and
the dataset manifest also pairs each target with its complete PAW+ACE
archive. The additional native-reference and replay checks are described in
[the PBE0 guide](vasp_pbe0.md).

### FNO width and Fourier modes

`fno_width` is the number of hidden feature channels at each grid point and
Fourier mode, not the number of retained modes. With `fno_width: 8`, the
correction maps three input fields (the MACE prediction, atomic-reference
potential, and geometry-derived atomic density envelope) to eight hidden
channels, processes them with `fno_layers` blocks, and returns one potential
correction. `model.channels` separately controls the MACE feature
multiplicities.

The spectral branch keeps the FFT wavevectors with `|q| < q_cutoff`, where
`q_cutoff` is in inverse angstroms (default 6.0). Wavevectors are computed
from the actual cell metric, so the cutoff is spherical in physical
reciprocal space, including for skew cells. The zero mode is included;
Nyquist planes of even-sized grids are excluded to preserve conjugacy.
There is no fixed mode-count setting. The retained count depends on:

- `q_cutoff`: increasing it admits shorter wavelengths.
- The cell: larger cells give more closely spaced reciprocal vectors.
- `grid_shape`: only frequencies represented on this grid are available.

For an orthogonal cell, the spacing along direction `i` is `2*pi/L_i`.
For an even grid size `N_i`, the largest retained axial index is `N_i/2 - 1`;
raising the cutoff cannot recover frequencies beyond the grid. Increasing
the cell size at fixed grid resolution therefore reduces the highest
representable physical frequencies. The grid and cutoff need to be checked
together when transferring to larger cells. The default cutoff is a model
setting, not a demonstrated convergence threshold for every material.

The learned radial filter is shared across modes and evaluated at each
physical `|q|`; adding modes does not add an independent set of parameters
per mode. The cutoff restricts the spectral mixing branch. It does not
remove the full-resolution MACE prediction or the local channel-mixing path.
`spectral_chunk_size` only controls how many retained modes are processed
together to limit memory use; it does not change the retained set.

### Sampling and optimization

`sampling.point_count` is the maximum number of sampled native-grid points
per structure (8192). `sampling.seed` is the base seed (70000), to which the
sample index is added. This sampling is independent of `training.seed`.
The data manifest supplies the training/validation/test split; this command
does not repartition it. Normalization uses only the training split.

The full PBE example lists all optimizer and stopping defaults. The
`backbone_lr` group contains the trainable MACE backbone parameters;
`head_lr` covers the decoder, PAW heads, and FNO when present. Both groups
are optimized jointly with Adam. The objective is the sum of field MSE and
active PAW-entry MSE, each divided by the squared training-set scale.
Scheduler patience counts validation checks, whereas early-stopping patience
counts epochs. `--epochs` is a maximum, and the best validation checkpoint
is retained. CPU runs use `device: cpu`; CUDA accepts `cuda` or `cuda:N`.

## Preparation, provenance, and outputs

`--dry-run` validates the recipe and prints all resolved settings without
creating files, reading datasets, or allocating a GPU. It does not certify
that reference data are complete. `--prepare-only` checks the data and writes
the cache without fitting. These options are mutually exclusive.

```sh
mapho-vasp train run --config train.yaml --prepare-only
mapho-vasp train model train TRAINING full
mapho-vasp train model evaluate TRAINING full --device cuda
```

The existing `train model train` command uses the saved training settings
when `--epochs` and `--device` are omitted. Explicit overrides are recorded
in the checkpoint and training report. Existing manifests without these
settings use the original defaults.

The output directory must be new; existing runs are never overwritten or
implicitly resumed. Each run contains:

- `config.input.yaml`: the exact original YAML bytes.
- `config.resolved.json`: defaults, CLI overrides, absolute paths, and the
  source-config hash.
- `manifest.json`: data contracts, normalization, architecture, optimizer
  settings, and hashes of the saved configuration files.
- `cache/`: the prepared training and evaluation inputs.
- `full/` or `local/`: `model.pt`, `history.json`, `training.json`,
  `test_fields.json`, and predicted arrays.

The checkpoint records the manifest hash and the effective optimizer
settings. Editing the saved configuration files is detected during training
or model loading. Make a new recipe and output directory for a new experiment.

PBE preparation still requires audited data and completed atomic references.
`target: pbe0_components` additionally performs the reference/replay audit and
checks the pairing and hashes of the ACE archive for each target. The YAML
interface does not bypass these checks or the convergence gate used to
generate the PBE0 dataset.
