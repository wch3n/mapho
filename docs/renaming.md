# MAPHO project rename

MAPHO means **Many-body Potential-based Hamiltonian Operator**. The canonical
project directory, Python package, distribution and CLI prefix are `mapho`.
The rename changes software identity, not the physical model or numerical data.

The migration completed on 16 October 2026 after the project queue was empty.
The canonical locations are:

- Project: `/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mapho`
- Runs: `/gpfs/scratch/acad/htbase/wchen/mapho/runs`

The project contains a `runs` symlink to scratch. The run data occupy about
136 GiB on scratch, with no duplicate run directory under the home project.
The renamed package is installed locally in both `.venv` and `.venv-qe`.

Keep the `mlham -> mapho` links under both the home project parent and the
scratch parent. Saved manifests, source snapshots, and virtual-environment
launchers still use those absolute paths. The links consume negligible space;
removing them would break those references. The source backup is in
`../.mapho-migration-v1/original_files`, and the original package is preserved in
`../.mapho-migration-v1/original_package`.

Compatibility is deliberate:

- `src/mapho` contains every implementation. A single lazy compatibility shim
  in `src/mlham/__init__.py` replaces the 46 per-module forwarding files.
  Old imports resolve to the identical canonical module and class objects;
  pickle class paths and `python -m mlham[.module]` remain supported.
- New data, checkpoints, and manifests use `mapho.*` schema identifiers.
  Readers normalize only the former `mlham.` prefix; format versions and
  physical validation remain strict. Historical files are read without
  rewriting their contents or hashes.
- New QE inputs use `prefix='mapho'`. Restart lookup reads the prefix from
  the corresponding `qe.in`, including old `mlham.save` calculations and
  restarts copied from an old PBE reference into a new PBE0 calculation.
- The validated VASP executable continues to use `MLHAM_EXPORT`, `MLHAM_IMPORT`,
  `MLHAM_PAW.dat` and its versioned matrix headers. The bridge's internal routine
  names and messages use MAPHO in newly patched copies; no binary rebuild is
  required to keep using the existing executable.
- Completed runs and immutable source snapshots are not rewritten. Their hashes
  retain their original meaning.

The source cleanup of 16 October 2026 also migrated the active manuscript
run drivers to `mapho`, included the compatibility helpers in isolated worker
snapshots, and removed stale forwarding modules from wheel build output.
The pre-cleanup source is backed up in `../.mapho-cleanup-20261016`; checks and
the change inventory are under `runs/code_cleanup_20261016`. The standalone
orbital baseline remains a supported module with tests and a CLI, rather than
being deleted because it predates the field model.

Validation after cleanup passed 216 CPU tests (one CUDA-only test skipped).
Separate processes using the original and cleaned source also gave bitwise
identical local fields, Fourier corrections, and PAW readouts for the saved
Si, LiF, and MoS2 checkpoints, using one held-out geometry per material on a
6x7x8 probe grid. Checkpoints and historical calculation files were not changed.

The migration bundle in `.mapho-migration-v1` beside the project records the
staged files, original hashes, validation results, and completed activation.
The activation checked both staged and original source hashes before changing
the project. The current manuscript and recent result files also passed a
114-file preservation check across the directory rename.

The subsequent [backend separation](backends.md) places QE and VASP routines
in `src/mapho/qe/` and `src/mapho/vasp/`. Central lazy aliases preserve the
previous flat imports. Shared models remain at the package root, and new
worker snapshots retain the nested directory layout.
