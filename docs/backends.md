# Package layout and commands

Reusable VASP code lives in `mapho.vasp`. Argument parsing is grouped by task
in `mapho.commands.vasp`. Research protocols and material-specific benchmarks
live in `mapho.experiments.vasp`; they are not a stable application API.

```text
src/mapho/
  vasp/                         Reusable VASP API
    paw.py                      PAW operator I/O, export, and replay
    model.py                    PAW model heads
    paw_conditioning.py         Conditioned equivariant PAW heads
    train.py                    Training, baseline loading, and prediction
    checkpoints.py              Repaired and joint checkpoint loading
    ace.py, exchange.py         ACE transport and independent construction
  commands/vasp/                Four groups of CLI handlers
    data.py                     Dataset and atomic-reference preparation
    train.py                    Model fitting and training studies
    predict.py                  Operator replay, band calculations, and plots
    validate.py                 Native checks and hybrid diagnostics
  experiments/vasp/             Research protocols and benchmark drivers
    *_dataset.py                Si, LiF, and MoS2 dataset protocols
    *_validation.py             Native validation studies
    *_bands.py, *_plot.py        Material-specific band checks and plots
    paw_repair.py, paw_joint.py  Repair and joint fine-tuning experiments
    activation_ablation.py      Activation comparison
    hybrid_pilot.py             PBE0 component diagnostic
  qe/                           Quantum ESPRESSO interfaces and workflows
  field_model.py                Shared MACE and Fourier field models
  fields.py, probes.py          Shared fields and probe operations
  exchange.py, ace_operator.py  Independent exchange and ACE implementations
  model.py, graph.py            Orbital-block baseline
  source_snapshot.py            Recursive worker snapshots and source hashes
```

The core VASP API does not import commands or experiments during inference:

```python
from mapho.vasp.paw import export_operator, prepare_replay
from mapho.vasp.train import load_model, predict
from mapho.vasp.checkpoints import load_repaired_model, load_joint_model
from mapho.qe.backend import QEConfig, run_reference
```

Use `mapho-vasp GROUP COMMAND` after installing the package. Each level has
`--help`, and selecting a group imports no electronic-structure or ML backend.

| Group | Commands |
|---|---|
| `data` | `atomic-reference`, `silicon`, `material`, `reference-chunk` |
| `train` | `run` (YAML), `model`, `pbe0-heads`, `material`, `paw-repair`, `paw-joint`, `activation` |
| `predict` | `paw`, `silicon-bands`, `binary-bands`, `silicon-plot`, `binary-plot` |
| `validate` | `paw`, `spin`, `training`, `ace`, `exchange`, `operator`, `hybrid`, `pbe0-convergence` |

```sh
mapho-vasp --help
mapho-vasp data material --help
mapho-vasp train model --help
mapho-vasp train run --config examples/training/pbe.yaml --dry-run
mapho-vasp predict paw export REFERENCE OPERATOR.h5
mapho-vasp validate ace --help
```

The equivalent module entry point is `python -m mapho.commands.vasp`.
Calculation workers use a group module, for example
`python -m mapho.commands.vasp.data silicon worker ROOT PHASE INDEX`.
Research protocols are also callable from Python through
`mapho.experiments.vasp.material_dataset.prepare` and related modules.

Legacy Python imports, forwarding entry points, `mlham-*` commands, and
`mapho-vasp-paw` have been removed. Use canonical backend imports and the
grouped dispatcher. The `mapho`, `mapho-qe`, and `mapho-fields` commands remain.
The QE command still supports `python -m mapho.qe`.

Experiment sources remain inside the package so new worker snapshots contain
all code needed by a study. Snapshots copy public Python modules recursively,
including `commands/` and `experiments/`. Manifests hash paths relative to the
package root, such as `experiments/vasp/silicon_dataset.py`. Existing snapshots
cannot be overwritten; resume a historical study using its own snapshot.

Saved numerical data and model weights are not rewritten. Readers still
recognize the former schema prefix. New native VASP calculations use
`MAPHO_EXPORT`, `MAPHO_IMPORT`, and `MAPHO_PAW.dat`; the private executable
must support that protocol. Historical full-object pickles that require removed Python module
names must use their original environment; current model loaders reconstruct
models from configuration and state dictionaries.

Private VASP patchers and Fortran bridge includes remain local and excluded
from worker snapshots and releases. Manuscript and benchmark artifacts also
remain local. See [release exclusions](releasing.md).
