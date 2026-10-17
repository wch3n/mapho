# Experimental PAW exchange interface

The private VASP adapter supplies the nonlocal exchange term missing from the
smooth-potential and PAW-D interface. It exports construction orbitals and
their **full Fock actions**, constructs compressed factors with the independent
`mapho.ace_operator.ACEOperator`, and imports these factors for a frozen
Hamiltonian diagonalization. Native ACE factors are optional validation data;
they are not the input to the independent factorization.

The scope is nonmagnetic, scalar PBE0 with a complex standard executable and
a fixed cell, FFT grid, reciprocal ordering, k mesh, and number of bands.
AEXX is 0.25, HFSCREEN is zero, and HFRCUT is zero. The private MPI extension
supports one rank per k-point group, with `KPAR` equal to the MPI rank count
and `NCORE=1`. Band and plane-wave distribution within a group, arbitrary-k
exchange, spin polarization, and SOC remain outside this interface.

## Operator and conventions

The replay solves the PAW generalized eigenproblem with the native kinetic
energy, native projectors and overlap, the imported smooth potential and
on-site D matrices, and the imported nonlocal exchange action. The full action
export is the nonlocal remainder matching those D matrices; adding an on-site
exchange term again would double count it.

The exported columns are Phi and W = K Phi in the complex plane-wave
coefficient basis. W includes the hybrid fraction and is a Hermitian
Hamiltonian action, not S^(-1) K Phi. PAW-normalized orbitals need not have
Euclidean-orthonormal columns. Factorization uses Phi† W, not an extra insertion
of the PAW overlap. VASP retains the native overlap in diagonalization.

MAPHO builds X from the full actions and applies `-X @ (X.conj().T @ c)`.
The private import branch uses an independently written GPU contraction;
it bypasses native ACE construction and native ACE contraction. Ordinary SCF
calculations without the import marker retain their native exchange path.

## Data flow

1. Run a converged native PBE0 reference with the private adapter. Alongside
   the existing `MAPHO_EXPORT` marker, `MAPHO_FOCK_EXPORT` requests final
   construction states and uncompressed actions in `MAPHO_FOCK.bin`.
   `MAPHO_ACE_EXPORT` additionally writes native factors for a comparison.
2. `mapho.vasp.ace.export_hybrid(reference, output)` checks the reference
   manifest, constructs MAPHO factors from the full actions, and packages
   their provenance with the smooth/PAW components. It preserves hashes of
   the source samples and component files.
3. `prepare_hybrid_replay(output, folder, potential=..., d=...)` prepares a
   frozen solve, optionally substituting predicted smooth and on-site fields.
   The default uses reference components. It writes `MAPHO_ACE_IMPORT`,
   imports the factors, and starts from fresh orbitals without a reference
   WAVECAR or CHGCAR. The private branch requires ICHARG=14 and NELMDL=0.

These Python functions require the existing prepared reference manifest and
native exported components. The data transport uses a MAPHO-defined format,
with explicit dimensions, lattice, k coordinates, reciprocal indices, and
complex128 arrays. Both Python and the private importer reject mismatched
metadata and malformed records.

### Parallel export and replay

Use a private build with the MPI extension for calculations with more than
one rank. For four GPUs, set `KPAR=4` and `NCORE=1` in the native reference
and launch four MPI ranks, one per GPU. The export collects the complete
orbital and action block from the owner of each k point. One rank writes the
archive in the same format used by the serial interface.

Prepare a matching frozen replay with:

```python
from mapho.vasp.ace import prepare_hybrid_replay

prepare_hybrid_replay("operator.h5", "replay", kpar=4)
```

Launch this directory with four MPI ranks and the same private executable.
The GPU path needs a CUDA-aware MPI transport. With the Open MPI 4 build in
the local NVHPC installation, the single-node launcher is:

```sh
cd replay
mpirun --bind-to none --mca pml ob1 --mca btl self,smcuda -np 4 /path/to/vasp
```

The `smcuda` transport handles GPU buffers during native communication
between k-point groups. The earlier serial launcher forced `self,vader`,
which does not support this multi-GPU path. Other MPI installations may use
a CUDA-enabled UCX transport. See the
[CUDA-aware Open MPI documentation](https://www.open-mpi.org/faq/?category=runcuda).

The default `kpar=1` preserves serial preparation. The manifest records the
required rank count, and the native bridge rejects layouts with more than
one rank per group. Each rank reads and checks the archive, then loads the
factors for its assigned k points. PAW matrices are replicated across groups
and must not be summed over those groups. This extension changes the
distribution of the calculation without changing the learned model or the
operator archive.

Keep `IBZKPT` with the native outputs. The band reader uses its reciprocal
coordinates when available, after checking the mesh and weights against
`vasprun.xml`. XML prints coordinates to eight decimal places, which loses
precision for fractions such as 1/6. Using `IBZKPT` preserves the native
coordinates for comparison with the binary exchange archive. Operator
exports record the hash of this file. The strict comparison between the
exchange mesh and the native band mesh remains unchanged.

The Python constructor is also available as:

```sh
mapho-vasp validate exchange MAPHO_FOCK.bin operator.ace.bin
```

`mapho.vasp_paw_ace.v2` identifies this combined archive. Partial hybrid
component archives remain distinct and cannot silently enter ordinary PBE
replay or training as a complete Hamiltonian.

## Validation

The driver `mapho-vasp validate ace` prepares an isolated source
snapshot and compares an unmodified native reference with the patched
reference, independently constructed factors, a fresh frozen replay, a
0.25 eV gauge-shift control, and a half-exchange negative control. It compares
factor **actions**, since unitary rotations of the factors do not change the
operator. The eigenvalue threshold is 0.1 meV over every requested band and
k point; construction-action comparisons use 1e-8 eV.

The completed A100 validation is recorded under
`runs/vasp_exchange_interface_v2/validation_v2` (job 16263619, 47 seconds,
16 October 2026). It uses primitive Si, 600 eV, a Gamma-centered 2x2x2 mesh,
and 12 construction bands: 96 eigenvalues in total.

| Check | Result |
| --- | --- |
| Patched versus unmodified native eigenvalues | Identical at the 1e-6 eV printed precision |
| Full-action reproduction by MAPHO ACE | Maximum column-norm error 8.39e-15 eV |
| MAPHO versus native ACE on construction states | Maximum difference 4.30e-15 eV |
| MAPHO versus native ACE on random probes | Maximum difference 2.13e-16 eV |
| Fresh frozen replay versus native eigenvalues | Identical at the 1e-6 eV printed precision |
| Uniform 0.25 eV shift, including the PAW overlap term | Exact shift at the printed precision |
| Half-exchange negative control | Maximum eigenvalue change 2.15 eV; detected |

Both native and replay sampled gaps are 2.437018 eV. This coarse-mesh value is
an interface check, not a converged Si PBE0 gap. The replay starts without
WAVECAR or CHGCAR and confirms the independent MAPHO contraction in its log.
`validation.json` and `audit.json` retain results, hashes, and provenance.
The initial failed attempt is retained in `validation`: VASP inserted a
mixing delay for random initial orbitals. The private patch now preserves
NELMDL=0 when importing an external operator; the ordinary path is unchanged.

The relevant CPU suite passed 78 tests, with its CUDA test skipped on CPU.
The earlier standalone A100 suite passed all 21 core tests. Synthetic transport
tests are in `tests/test_vasp_ace.py`; they require no licensed inputs.

## Limitations and release boundary

ACE reproduces the supplied full actions on the construction subspace.
It is not exact on arbitrary trial states or at new k points. Changed occupied
orbitals require new full exchange actions and rebuilt factors. Passing a
frozen native replay therefore validates the interface, not the predictive
accuracy of a learned hybrid model or an exchange speedup.
Constructing ACE requires exchange actions on the full construction block.
A fast application of the compressed operator does not remove this initial
cost. Any timing of hybrid inference must include preparation of its inputs,
construction of exchange, factorization, and final diagonalization. In this
pilot the factors come from the converged hybrid reference, so their reuse
does not establish an SCF-free hybrid prediction. Predicting exchange actions
or factors would require a separate model and validation of its target subspace.
The replay targets eigenvalues of the frozen generalized eigenproblem;
hybrid total energies, forces, stress, and a learned hybrid SCF loop are outside
this implementation and are not certified by its convergence marker.

No PBE0 training is performed by this driver. The present MAPHO field and PAW
heads still require a learned exchange representation for a structure-only
hybrid model. Using reference exchange while substituting learned local/PAW
components remains a conditional prediction.

All source modifications and Fortran bridge files remain private. Packaging
excludes the patchers, bridge sources, build tree, run data, pseudopotentials,
and executables from both wheels and source distributions. The public Python
transport and independent ACE core contain no VASP source. See
[the release checks](releasing.md); no release is published by validation.
