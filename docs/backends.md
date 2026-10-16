# Backends, commands, and workflows

VASP integration has three separate layers. The reusable API handles operator
files, models, training, prediction, and checkpoint loading. Command modules
parse arguments and call that API or a workflow. Workflows prepare datasets,
launch calculations, validate results, or reproduce specific experiments.

```text
src/mapho/
  vasp/                         Reusable VASP API
    paw.py                      PAW operator I/O, export, and replay
    model.py                    PAW model heads
    paw_conditioning.py         Conditioned equivariant PAW heads
    train.py                    Training, baseline loading, and prediction
    checkpoints.py              Repaired and joint checkpoint loading
    ace.py, exchange.py         ACE transport and independent construction
  commands/vasp/                Argument parsing and dispatch
    __init__.py                 mapho-vasp command selection
    paw.py, train.py, ...        Thin per-command interfaces
  workflows/vasp/               Calculation and experiment orchestration
    *_dataset.py                Reference dataset preparation and sampling
    *_validation.py             Native validation workflows
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
  _compat.py                    Historical module and command aliases
```

The core VASP API does not import the workflow or command packages during
normal use. Conditioned checkpoints can be loaded directly:

```python
from mapho.vasp.paw import export_operator, prepare_replay
from mapho.vasp.train import load_model, predict
from mapho.vasp.checkpoints import load_repaired_model, load_joint_model
from mapho.qe.backend import QEConfig, run_reference
```

Use `mapho-vasp` to select a command after installing the package:

```sh
mapho-vasp --help
mapho-vasp paw --help
mapho-vasp train --help
mapho-vasp material-dataset --help
mapho-vasp paw-joint --help
```

The equivalent module entry point is `python -m mapho.commands.vasp`. Individual
commands also work as modules, for example
`python -m mapho.commands.vasp.silicon_dataset --help`. Worker subprocesses use
these command modules explicitly. Import workflow functions when orchestrating
from Python, for example `mapho.workflows.vasp.material_dataset.prepare`.

The existing `mapho-vasp-paw` and `mlham-vasp-paw` console commands remain valid.
Historical imports such as `mapho.vasp.silicon_dataset`,
`mapho.vasp_silicon_dataset`, and `mlham.vasp_silicon_dataset` resolve to the
same workflow module. Their old `python -m` commands dispatch to the new CLI
without executing a second copy of the workflow. Old conditioned-model loader
names re-export the core checkpoint functions. The small `main()` delegates
in the core PAW, exchange, and training modules preserve existing entry points;
all argument parsing lives in the command layer.

The QE package is unchanged by the VASP workflow separation. It provides
`mapho.qe.backend`, `mapho.qe.exchange`, `mapho.qe.files`, and `mapho.qe.upf`,
alongside QE reference and validation workflows. The existing `mapho-qe` and
`python -m mapho.qe` commands remain valid.

New worker snapshots copy the public Python tree recursively, including
`commands/` and `workflows/`. Manifests hash paths relative to the package root,
such as `workflows/vasp/silicon_dataset.py`. Existing snapshots cannot be
overwritten. Historical runs retain their recorded files and hashes; resume
those studies with their own source snapshot. Data schemas, model parameters,
and the validated VASP file protocol are unchanged.

Private VASP patchers and Fortran bridge includes remain local and excluded
from worker snapshots and releases. Manuscript and benchmark artifacts also
remain local, as requested. See [release exclusions](releasing.md).

Validation of this separation passed 317 CPU tests, with one CUDA-only test
skipped. Saved Si, LiF, and MoS2 checkpoint outputs were bitwise identical
before and after the move for one held-out structure per material on a 6x7x8
probe grid. The comparison covered the local field, Fourier correction, and
PAW readout. Wheel and source-archive exclusion checks also passed. The local
validation record is `runs/workflow_layout_20261016/refactor.json`.
