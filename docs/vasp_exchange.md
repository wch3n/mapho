# Experimental PAW exchange interface

The private VASP adapter supplies the nonlocal exchange term missing from the
smooth-potential and PAW-D interface. It exports construction orbitals and
their **full Fock actions**, constructs compressed factors with the independent
`mapho.ace_operator.ACEOperator`, and imports these factors for a frozen
Hamiltonian diagonalization. Native ACE factors are optional validation data;
they are not the input to the independent factorization.

The initial scope is nonmagnetic, scalar PBE0, a complex standard executable,
one MPI rank, and a fixed cell, FFT grid, reciprocal ordering, k mesh, and number
of bands. AEXX is 0.25, HFSCREEN is zero, and HFRCUT is zero. The implementation
is not yet an arbitrary-k, spin-polarized, SOC, or distributed interface.

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
metadata and malformed records. The Python constructor is also available as:

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
