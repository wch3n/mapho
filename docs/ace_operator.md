# Standalone ACE numerical core

`mapho.ace_operator` implements exchange compression using the public equations
of Lin Lin, *Adaptively Compressed Exchange Operator*, J. Chem. Theory Comput.
**12**, 2242–2249 (2016), [DOI](https://doi.org/10.1021/acs.jctc.6b00092),
[public manuscript](https://arxiv.org/abs/1601.07159), Eqs. (10)–(15).

Given construction orbitals Phi and the actions W = K Phi of a fixed negative
exchange operator, the module constructs

    M = Phi† W,  -M = L L†,  X = W L^(-†),  K_ACE v = -X (X† v).

The factorization uses a triangular solve. The retained factors have shape
`(basis_size, number_of_construction_states)`; no dense operator is needed.
Hermiticity, the sign of the Gram matrix, and reproduction of the supplied
actions are checked. Singular construction sets are rejected without adding
a diagonal shift. Real and complex tensors are supported on the supplied
PyTorch device. The hybrid fraction must already be included in W.

```python
from mapho.ace_operator import ACEOperator

# phi and full_exchange_action are supplied by a separate, authorized backend.
actions = full_exchange_action(phi)
ace = ACEOperator.from_samples(phi, actions)
compressed_actions = ace.apply(trial_orbitals)
```

## Scope

The numerical core is called by validation and by the experimental private
VASP exchange adapter, which constructs factors from exported full actions.
MAPHO training does not yet call it. The exchange-learning branch still
uses the full FFT Fock operator. Compression is most useful for repeated
applications of a fixed operator during diagonalization; changing the learned
exchange gate or its reference orbitals requires rebuilding the factors.

Exact reproduction applies to the span of the construction orbitals at a fixed
exchange operator. It is not guaranteed for arbitrary empty bands or new
k-points. New construction states require new full exchange actions. Updating
the occupied orbitals in an SCF cycle also requires rebuilding the compression.

The ambient representation must use the Euclidean inner product. Grid
quadrature weights or a nonorthogonal overlap cannot simply be ignored.
This module supplies no PAW metric transformation, augmentation terms, Fock
integrals, Coulomb singularity treatment, SCF driver, or file-format adapter.
It is not a complete PBE0 Hamiltonian or a replacement for a VASP interface.

## Provenance and validation

This module and its synthetic tests were newly written from the cited
mathematics. They do not import the existing VASP adapters or use VASP source,
internal data formats, pseudopotentials, executables, or ACE factors.
The assistant had previously accessed VASP source in this conversation;
therefore this work is not represented as a formal clean-room process, and
this note does not establish compliance of earlier work or of the repository
as a whole. The separate private VASP bridge is excluded from public releases.

Run the focused checks with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_ace_operator.py
```

The checks use synthetic negative Hermitian matrices and cover interpolation,
Hermiticity, negativity, construction-basis changes, hybrid scaling, invalid
inputs, and the limitation outside the construction space. A CUDA comparison
runs only when a GPU is available.

## Validation against QE outputs, 16 October 2026

CPU validation passed on 16 structures per benchmark: Si–HSE, LiF–HSE, and
LiF–PBE0. The LiF geometries are shared between the two functionals, so these
are 48 structure–functional cases, not 48 independent geometries. Every case
contains eight k points and eight construction bands: 384 k-point blocks and
3,072 construction states in total. All calculations below use complex128.
This is a numerical validation of compression, not a learned-model benchmark.

| Benchmark | Maximum construction-action error (eV) | Maximum difference from QE ACE actions on construction states (eV) | Eight-state Ritz maximum error (eV) |
|---|---:|---:|---:|
| Si–HSE | 1.65e-14 | 3.29e-14 | 3.20e-14 |
| LiF–HSE | 1.04e-13 | 1.34e-13 | 9.95e-14 |
| LiF–PBE0 | 8.13e-14 | 1.12e-13 | 8.88e-14 |

Factors were constructed from the saved **full exchange actions**, never from
QE's ACE factors or ACE actions. Two random probes per k point, orthogonal to
the eight construction bands, give action RMSEs of 134.80, 261.88, and 261.78
meV, respectively. These agree with the saved QE ACE results. Enlarging the
construction set to include both probes reduces their maximum action error
to 1.26e-15 eV across all cases. An occupied-only construction also fails to
reproduce the omitted empty states; its maximum action error reaches 1.42 eV.
Thus the checks establish the expected subspace limitation as well as accuracy.

An additional full FFT exchange calculation was performed on the first
training structure of each benchmark, including all eight k points, all eight
bands, and both probes. Maximum differences from native QE were 8.94e-11 eV
(Si–HSE), 1.67e-12 eV (LiF–HSE), and 1.68e-12 eV (LiF–PBE0). The unscreened
Coulomb singularity correction was computed independently from the cell,
cutoffs, and k mesh. ACE constructed from these independently calculated
actions also passed the 1e-8 eV comparison threshold.

The real-material eigenvalue check is an eight-state generalized Ritz problem
that keeps the exact non-exchange Hamiltonian in that subspace. It is not a
full plane-wave band calculation. Separate 125-dimensional controlled tests
use screened and unscreened FFT exchange with a fixed density matrix and a
nondegenerate one-body Hamiltonian. Starting with a random six-state subspace,
adaptive compression converges in eight rebuilds to the lowest six eigenvalues
of the full dense Hamiltonian, with maximum errors of 9.95e-14 and 6.13e-14 eV.
This tests repeated compression and diagonalization, not a hybrid SCF cycle.

The CPU suite passed 33 tests, with the CUDA-specific test skipped. The CPU
debug-node job (16262105) completed successfully. The A100 job (16262073) also
passed all numerical checks, including the three independent FFT comparisons.
The maximum construction-action errors on the A100 are 6.62e-15, 2.77e-14,
and 2.29e-14 eV for Si–HSE, LiF–HSE, and LiF–PBE0, respectively. The largest
CPU–GPU action difference is 1.09e-13 eV. The CUDA unit test also passed.

Machine-readable results and input/source hashes are in
[`results.json`](../runs/ace_validation_v1/cpu/results.json), with the
[unit-test report](../runs/ace_validation_v1/cpu_unit_tests.xml).
The [A100 results](../runs/ace_validation_v1/results.json) retain the device,
source hashes, individual comparisons, and allocated CUDA memory.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_ace_operator.py \
  tests/test_ace_physical.py tests/test_exchange.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m mapho.qe.ace_validation --runs runs \
  --output runs/ace_validation_new --device cuda --require-gpu-name A100
```

## VASP integration boundary

The existing scalar-potential and PAW interface does not supply the nonlocal
exchange action needed to replay a complete hybrid Hamiltonian. Integrating
this standalone ACE into VASP requires a separately validated bridge for
exporting construction states and full exchange actions, and for applying the
resulting operator with consistent PAW terms and no double counting. Selecting
VASP's built-in ACE does not load the Python operator. The public local-potential
plugin supplies a grid potential rather than a general nonlocal-operator hook.
See the public [LFOCKACE documentation](https://vasp.at/wiki/LFOCKACE) and
[local-potential plugin documentation](https://vasp.at/wiki/PLUGINS/LOCAL_POTENTIAL).

Those standalone checks do not validate a VASP bridge. The separate
[experimental exchange interface](vasp_exchange.md) now exports full exchange
samples, builds factors with this module, and applies them through a private
Fortran contraction. Its native eigenvalue controls are recorded separately.
Private VASP patch material remains excluded from release artifacts as
described in [the release guide](releasing.md).
