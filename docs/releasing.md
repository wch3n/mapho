# Release exclusions

The local VASP patch scripts (`src/mapho/vasp/vasp*_patch.py`) and Fortran bridge files are
private integration material. Keep them on the licensed installation and out
of public releases. Local build trees and calculation files under `runs` are
also private. They are not dependencies of the standalone ACE core.

Python wheels exclude patch modules through `setup.py`. Package data are
disabled, and the build hook removes private files and obsolete flat modules
from stale build output. These exclusions also cover nested backend packages.
`MANIFEST.in` excludes the same material from source distributions.
`.gitignore` prevents accidental addition of these files to a new repository;
it does not untrack files already committed. `.gitattributes` also excludes
them from `git archive`. Never publish a recursive copy of the working tree
or follow the `runs` symlink when assembling a release.

The SHA-256 helper used by the public PAW interface lives in `file_utils.py`;
ordinary imports and reproducible workflow snapshots no longer require the
private patch module. Source-distribution tests skip the private patch-helper
test when that module is absent. Existing local scripts and bridges are kept.

Build with the standard Python packaging frontend, then check both artifacts:

```sh
python -m build
python tools/check_release.py dist/*.whl dist/*.tar.gz
```

The checker fails on private patch and bridge filenames, Fortran sources,
pseudopotential files, VASP executables, calculation trees, and archive links.
It reads archive member names and metadata only. This is an enforceable check
for the known private files, not a certification of the provenance of every
line of the remaining Python code. Review any new source-derived material
before adding it to a release. No publication is performed by these commands.
