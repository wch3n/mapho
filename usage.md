# Usage

MAPHO provides separate interfaces for VASP PAW operators, Quantum ESPRESSO
fields, and an independent orbital-block model. The [tutorial](tutorial.md)
follows the VASP PBE route from atomic structures to predicted bands.

## Contents

- [Installation](#installation)
- [VASP commands](#vasp-commands)
- [Training from YAML](#training-from-yaml)
- [Band inference](#band-inference)
- [Other interfaces](#other-interfaces)
- [Development](#development)

## Installation

Use Python 3.10 or newer in a dedicated environment. For a new checkout:

```bash
git clone https://github.com/wch3n/mapho.git
cd mapho
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[references,fields,test]' matplotlib
```

The core dependencies include PyTorch, MACE, e3nn, ASE, NumPy, h5py, and
PyYAML. The `references` extra supplies pymatgen for input generation. The
`fields` extra supplies SciPy for field operations. Matplotlib is used by
the band example. The CUDA build of PyTorch and the runtime of VASP must
match the compute environment.

An existing environment can be used if these dependencies are already
available. Preserve the recorded versions when reproducing a checkpoint.
For the existing project environment on the cluster, invoke
`env -u PYTHONPATH .venv/bin/python` or
`env -u PYTHONPATH .venv/bin/mapho-vasp` directly. This avoids an unrelated
`PYTHONPATH` or an activation script containing the former project path.

VASP calculations require a licensed installation with the private MAPHO
export/import interface and local PBE PAW datasets. The public repository
does not contain these files. The examples use `MAPHO_EXPORT`,
`MAPHO_IMPORT`, and `MAPHO_PAW.dat`. An ordinary unmodified executable cannot
perform the operator export and replay described here.

## VASP commands

```bash
mapho-vasp --help
mapho-vasp data --help
mapho-vasp train --help
mapho-vasp predict --help
mapho-vasp validate --help
```

| Group | Purpose |
|---|---|
| `data` | Generate structures, calculate targets, and prepare atomic references |
| `train` | Fit potential and PAW corrections, including YAML recipes |
| `predict` | Export/replay operators and run material-specific band studies |
| `validate` | Check native interfaces, operator reconstruction, and reference sampling |

The equivalent Python entry point is `python -m mapho.commands.vasp`.
Reusable APIs are in `mapho.vasp`. Material-specific protocols are in
`mapho.experiments.vasp`. See [the package layout](docs/backends.md).

The data generators for Si, LiF, and MoS2 use the prepared metadata and native
operator files. They do not import arbitrary folders containing ordinary
VASP outputs. The [tutorial](tutorial.md#1-data-preparation) describes the
complete Si workflow. The same atomic-reference commands apply to the LiF
and MoS2 PBE datasets.

Execution commands run within an existing scheduler allocation. They do not
submit jobs. The current dataset drivers use one MPI rank per VASP process
and distribute independent structures across the visible GPUs. Use A100
allocations with the runtime of the selected executable. Keep calculation
and training output on scratch storage.

## Training from YAML

Start from [the PBE recipe](examples/training/pbe.yaml), set the paths, and
inspect the resolved configuration:

```bash
mapho-vasp train run --config train.yaml --dry-run
mapho-vasp train run --config train.yaml
```

`mode: full` selects MAPHO-FNO and `mode: local` selects MAPHO-MACE. Both
predict a potential correction and PAW matrix corrections. The configuration
of the model is stored in the checkpoint and reconstructed at inference.

YAML paths are relative to the configuration file. CLI overrides take
precedence over YAML values, followed by the defaults. For example:

```bash
mapho-vasp train run --config train.yaml --epochs 600 --seed 47 \
  --mode local --output runs/mace_seed47
```

`--dry-run` checks the recipe without reading the dataset or creating a run.
To inspect data preparation separately, use:

```bash
mapho-vasp train run --config train.yaml --prepare-only
mapho-vasp train model train /absolute/path/to/training full
mapho-vasp train model evaluate /absolute/path/to/training full
```

After `--prepare-only`, use the latter two commands. Running `train run` again
with the same output path would attempt to create that directory again.
Output directories must be new. Existing runs are not implicitly resumed.

The [training guide](docs/training.md) defines `num_interactions`, `lmax`,
`fno_width`, `q_cutoff`, grid settings, and the data contract. The
[PBE0 recipe](examples/training/si_pbe0.yaml) requires its separate reference
and exchange checks, described in [the PBE0 guide](docs/vasp_pbe0.md).

## Band inference

For a single Si checkpoint produced by the YAML workflow, use:

```bash
python examples/vasp/si_bands.py \
  --training /absolute/path/to/training \
  --mode full --output /absolute/path/on/scratch/si_bands
```

The example predicts primitive Si from an atomic reference and compares its
bands with a converged PBE operator on X–Gamma–L. It checks primitive-cell
sampling and native operator replay before writing the comparison. It uses
the executable and PAW dataset recorded during training. See the
[inference section](tutorial.md#3-inference-and-band-structure) for the
calculation sequence and output files.

The older `predict silicon-bands` and `predict binary-bands` commands are
study drivers with additional artifact requirements. They are not generic
entry points for a newly trained single model. The standalone example uses
the public operator and model APIs without those historical dependencies.

For a prepared atomic operator, the essential inference calls are:

```python
from mapho.vasp.train import load_model, predict
from mapho.vasp.paw import prepare_replay, sha

model = load_model(training_directory, "full", "cuda")
potential, d = predict(
    model, {"baseline": atomic_h5, "baseline_sha256": sha(atomic_h5)}, "cuda"
)
prepare_replay(
    atomic_h5, band_directory, potential=potential, d=d, kpoints=kpoints
)
```

Here, `kpoints` is an array of fractional reciprocal coordinates with shape
`(n, 3)`. `prepare_replay` writes inputs for VASP. It does not execute the
eigensolver. The Python checkpoint is evaluated by MAPHO, and VASP reads
the resulting potential and PAW matrices.

## Other interfaces

- `mapho-qe` and `mapho-fields` provide the QE field route. See
  [the backend guide](docs/qe.md) and [field learning](docs/field_learning.md).
- `mapho` provides the independent orbital-block model. Its demonstration
  uses synthetic labels and requires no DFT executable:

```bash
mapho demo --output runs/orbital_demo --epochs 80
```

The orbital model has its own [data conventions](docs/orbital_baseline.md).
Its NPZ samples and checkpoints are distinct from the PAW HDF5 operators.

## Development

```bash
python -m pip install -e '.[test,dev]'
python -m pytest -q
ruff format --check .
```

Use `ruff format .` to apply the configured Python style. Run computation
tests on allocated nodes in a shared cluster. Unit tests use synthetic data
and do not establish accuracy on an unseen material.

Follow [the release guide](docs/releasing.md) when building distributions.
Private VASP integration files, native inputs, run directories, and manuscript
artifacts are excluded from public releases.
