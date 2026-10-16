# MAPHO project name and stored data

MAPHO means **Many-body Potential-based Hamiltonian Operator**. The Python
package, distribution, and command prefix are `mapho`. The project is at
`/gpfs/home/acad/ucl-modl/wchen/mxene_proj/mapho`; its `runs` link points to
`/gpfs/scratch/acad/htbase/wchen/mapho/runs`.

The rename and package cleanup change software organization without changing
the physical model or stored numerical results. Active code uses canonical
imports from `mapho`, `mapho.qe`, and `mapho.vasp`. The old `mlham` package,
flat backend aliases, and `mlham-*` commands have been removed. VASP commands
are grouped under `mapho-vasp`; see the [package layout](backends.md).

Stored calculation data retain their existing contracts:

- New serialized data use `mapho.*` schema identifiers. Readers recognize the
  former `mlham.` prefix while validating the format and version. Existing
  data and checkpoints are read without rewriting their contents or hashes.
- QE restart lookup reads the prefix from `qe.in`, including calculations
  whose restart folder is named `mlham.save`.
- New VASP calculations use `MAPHO_EXPORT`, `MAPHO_IMPORT`, `MAPHO_PAW.dat`,
  and `MAPHO_PAW_V1`/`MAPHO_PAW_V2` text headers. This requires the matching
  private executable; changing only the marker names would not enable export
  in an older binary. Existing raw `MLHAM_*` files retain their saved protocol.
  Already exported HDF5 operators can be read directly and replayed through
  the new interface.
- Completed runs and immutable source snapshots retain their original files
  and hashes. Resume old studies with their saved sources. Any filesystem
  links needed by recorded absolute paths are independent of Python aliases.

Current model loaders reconstruct models from configuration and state
dictionaries. Historical full-object pickles referring to removed module
names require their original source environment. Model loading should use
`mapho.vasp.train` or `mapho.vasp.checkpoints` directly.

The backups beside the project and validation records under `runs` document
successive cleanup steps. Manuscript files, benchmark artifacts, private VASP
patchers, and Fortran bridges remain local and excluded from publication.

The native `MAPHO_*` protocol was checked in
`runs/vasp_mapho_protocol_v1/validation` (A100 job 16272462). Scalar Si PBE0
export and independent ACE replay match the unmodified executable for all
96 eigenvalues at the printed precision of 1e-6 eV. The 0.25 eV gauge shift
and reduced-exchange negative control also pass. Existing studies retain
their original protocol and source snapshots.
