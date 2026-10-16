# Backend packages

The electronic-structure interfaces live in separate Python packages:

```text
src/mapho/
  qe/
    backend.py             QEpy field export and frozen-potential solves
    exchange.py            QE exchange reference generation
    files.py, upf.py        QE restart lookup and UPF metadata
    field_dataset.py       QE field datasets
    supercell_reference.py Supercell reference calculations
    kpoint_convergence.py  Sampling convergence
    kpoint_recovery.py     Restart and recovery workflows
    ace_validation.py      ACE checks against saved QE outputs
  vasp/
    paw.py                 PAW operator export and replay
    model.py, train.py     PAW model heads and training
    exchange.py, ace.py    Fock export and independent ACE integration
    *_dataset.py           VASP reference datasets
    *_validation.py        Native validation workflows
    *_bands.py, *_plot.py  Band calculations and plotting
    paw_*.py               PAW conditioning, repair, and joint fitting
  field_model.py           Shared MACE and Fourier field models
  fields.py, probes.py     Shared field representations and probe operations
  exchange.py             Independent exchange operator
  ace_operator.py         Independent ACE implementation
  model.py, graph.py       Orbital-block baseline
  source_snapshot.py      Recursive worker snapshots and source hashes
  _compat.py              Lazy aliases for historical backend imports
```

For example:

```python
from mapho.qe.backend import QEConfig, run_reference
from mapho.vasp.paw import export_operator, prepare_replay
from mapho.vasp.train import load_model, predict
```

Commands follow the same package layout:

```sh
python -m mapho.qe --help
python -m mapho.qe.exchange --help
python -m mapho.vasp.paw --help
python -m mapho.vasp.material_dataset --help
```

The console commands `mapho-qe` and `mapho-vasp-paw` are unchanged. Historical
imports such as `mapho.vasp_paw`, `mapho.qe_exchange`, and their `mlham`
equivalents resolve lazily to the same canonical modules. Old checkpoint
class paths and command lines remain supported without duplicate source files.
`from mapho.qe import QEConfig` remains available through the QE package.
Data schema identifiers and the validated VASP file protocol are unchanged.

New calculation workers copy the public Python package recursively, retaining
`qe/` and `vasp/`. Manifests record paths relative to the package root, such as
`vasp/paw.py`, with SHA-256 hashes. Snapshots cannot overwrite existing source
directories. Historical runs and their snapshots retain their original files
and hashes; resume them with their own recorded source snapshot.

The private patchers `vasp/vasp_patch.py`, `vasp/vasp_ace_patch.py`, and their
Fortran bridge includes remain local. Calculation snapshots, wheels, and
source distributions exclude them. See [release exclusions](releasing.md).

Validation of the separation on 16 October 2026 passed 267 CPU tests; one
CUDA-only test was skipped. The saved Si, LiF, and MoS2 checkpoints produced
bitwise-identical local fields, Fourier corrections, and PAW readouts before
and after the move, evaluated on one held-out structure per material with a
6x7x8 probe grid. Both Python environments passed import checks. Wheel and
source archives passed the private-file exclusions. The local validation
record is `runs/backend_layout_20261016/refactor.json`.
