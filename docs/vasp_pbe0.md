# PBE0 extension: native component pilot

The present MAPHO PAW model predicts a smooth potential and on-site D matrices.
For PBE this completes the operator with native kinetic energy, projectors and
overlap. PBE0 additionally requires nonlocal Fock exchange. An FNO that outputs
a scalar potential does not by itself supply that orbital-dependent operator.
The existing QE exchange-gate experiments are a separate implementation; their
norm-conserving exchange action cannot be inserted unchanged into a PAW model.

## First test

`runs/vasp_si_pbe0_components_v2` contains a primitive-Si diagnostic, submitted
on 2026-10-16 as array job **16261823**, with one A100 per mesh. The two meshes
can execute independently. All run files and private PAW inputs reside on
scratch. The lattice constant is 5.43 angstrom, matching the existing Si
benchmark; ENCUT is 600 eV and NBANDS is 12. The Gamma-centered 2x2x2 and 3x3x3
meshes test the interface and sensitivity, not converged bulk PBE0 accuracy.

At each mesh the workflow performs:

1. Native PBE SCF with exact diagonalization, followed by replay of its
   complete exported operator.
2. Native PBE0 SCF initialized from the same-mesh PBE orbitals, with the density
   and orbitals free to relax.
3. A continuation of the PBE0 calculation with the unmodified executable, to
   check restart stability and agreement with the bridge-enabled calculation.
4. A diagnostic solve using the exact exported PBE0 smooth potential and PAW
   D matrices, with the remaining Fock action deliberately omitted.

PBE0 uses `LHFCALC=.TRUE.`, `AEXX=0.25`, `HFSCREEN=0`, and `HFRCUT=0`: there is
no HSE screening or finite real-space exchange cutoff. No reduced exchange
mesh is requested. The first reference uses the direct minimizer
`ALGO=Damped`, `TIME=0.4`, and `LFOCKACE=.FALSE.`. This is a conservative native
reference, not an exchange application speed benchmark. The full hybrid SCF
is never run with frozen density. These choices follow the
[VASP hybrid solver documentation](https://vasp.at/wiki/LHFCALC) and the
[orbital requirements for hybrid calculations](https://vasp.at/wiki/Band-structure_calculation_using_hybrid_functionals).

Both native interface controls must agree to 0.1 meV across the requested
bands. The output records absolute and VBM-aligned frontier errors, sampled
gaps, complete band tables, input/executable/source hashes and wall times.
`completion.json` is only written when both meshes finish and their controls
pass. The omitted-Fock calculation is a diagnostic of the missing physical
term. It is neither an ML prediction nor a lower bound on a band-fitted
effective Hamiltonian.

## Distinguishing partial data from a Hamiltonian

The default `export_operator` rejects hybrid references. An explicit
`components_only=True` exports the available smooth/PAW arrays under the
separate `mapho.vasp_paw_hybrid_components.v1` schema, marked
`complete_hamiltonian=false`. The ordinary PBE training data schema remains
unchanged and does not accept these hybrid component files.

`prepare_replay(..., omit_fock=True)` is required to run the diagnostic. It
preserves the exported arrays, disables hybrid exchange and freezes those
arrays with the existing native import path. Without that explicit request,
replay rejects partial hybrid data, including legacy files incorrectly marked
as complete. No hybrid wavefunctions are passed to the omitted-Fock solve.

The raw hybrid D matrices are preserved in the component archive. The export
records their maximum antisymmetry and rejects discrepancies above 1e-6 eV;
the PBE and overlap-matrix checks retain their original 1e-10 tolerance. The
diagnostic replay uses `(D + D.T)/2` and records the maximum change. This
bounded numerical projection is only applied to the explicitly requested
hybrid diagnostic, never to learned matrix predictions.

The retained v1 attempt found one unconverged upper PBE band (0.626 meV replay
difference) and native PBE0 on-site antisymmetry of 1.42e-7 eV. Its failed
control and the raw native outputs are retained. The v2 changes tighten the
PBE diagonalization and explicitly account for that small hybrid numerical
asymmetry; neither issue is hidden by loosening the eigenvalue threshold.

The driver is `python -m mapho.commands.vasp.hybrid_pilot`. Preparation stores an isolated
source snapshot; execution checks it and the executable hashes. The GPU job
uses this snapshot so subsequent code changes do not alter an active run.

## Completed diagnostic

Both v2 array tasks completed on 2026-10-16, in 1 min 37 s and 4 min 54 s.
The PBE replay controls differ by at most 0.001 and 0.005 meV, respectively.
The PBE0 bridge/reference and unmodified native continuations differ by at
most 0.001 meV at either mesh. All comparisons cover 12 bands at every mesh
point. The maximum change from making D Hermitian is below 7.12e-8 eV per
matrix entry, recorded separately for each case.

| Mesh | PBE sampled gap (eV) | PBE0 sampled gap (eV) | PBE0 components with Fock omitted (eV) |
| --- | ---: | ---: | ---: |
| 2x2x2 | 0.59 | 2.44 | -0.15 |
| 3x3x3 | 0.69 | 2.21 | 0.05 |

A negative value denotes overlap between the selected occupied and empty
band manifolds. For the Fock-omitted calculation, the frontier MAE after VBM
alignment is 1.286 and 1.118 eV, respectively, using the two highest occupied
and two lowest empty bands at every k point. Thus omission affects relative
band energies as well as their common energy reference. These are errors of
an intentionally incomplete native operator, not trained MAPHO errors.

The sampled PBE0 gaps still differ by 0.230 eV between meshes; these numbers
must not be quoted as converged Si band gaps. The source and native output
identities are checked in `audit.json`; `completion.json` and per-case
`bands.csv` retain the full numerical results. Twenty-six relevant Python
tests pass, including rejection of incomplete hybrid Hamiltonians, preservation
of raw D, the bound on Hermitian projection, and the original scalar/spin
export/replay tests.

## Training after the interface test

A complete PBE0 model needs an exchange representation in addition to the
current field and PAW heads. The [experimental private exchange interface](vasp_exchange.md)
exports full Fock actions and imports independently constructed ACE factors,
including the matching PAW partition. Its unlearned replay must reproduce
native PBE0 before fitting. Keeping native
exchange would allow the current MACE and GELU/FNO branches to learn the local
and on-site pieces, but would still require reference orbitals and would not
establish structure-only or SCF-free hybrid inference.

Once that interface and PBE0 k-point convergence pass, use paired geometries
and structure splits to compare MAPHO-MACE and MAPHO-FNO. Report errors of
the learned local/on-site components and exchange separately, followed by a
complete native eigenvalue comparison. The present pilot starts no PBE0
training and makes no claim about its predictive accuracy.
