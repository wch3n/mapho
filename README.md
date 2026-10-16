# MAPHO

**Many-body Potential-based Hamiltonian Operator**

MAPHO learns the electronic Hamiltonian through a real-space potential and,
for PAW calculations, on-site matrix corrections. A MACE backbone describes
the local atomic environments. An optional Fourier neural operator (FNO)
couples the resulting fields across the periodic cell. The predicted operator
is then diagonalized with the kinetic energy, projectors, and overlap of the
DFT backend.

## Contents

- [Theory](#theory)
- [Getting started](#getting-started)
- [Documentation](#documentation)
- [Scope](#scope)

## Theory

For the VASP PAW implementation, the eigenproblem is

$$
\widetilde H\,\widetilde\psi_{n\mathbf k}
=\epsilon_{n\mathbf k}S\,\widetilde\psi_{n\mathbf k},
\qquad
\widetilde H=T+\widetilde V+
\sum_{aij}|p_i^a\rangle D_{ij}^a\langle p_j^a|.
$$

Here, $T$ is the kinetic operator, $\widetilde V$ is the smooth local
potential, $p_i^a$ are the PAW projectors, and $S$ is the overlap operator.
MAPHO learns corrections to a reference obtained from superposed atomic
densities:

$$
\widetilde V^{\mathrm{ML}}(\mathbf r)
=\widetilde V^{(0)}(\mathbf r)
+\Delta V_{\mathrm{MACE}}(\mathbf r)
+\Delta V_{\mathrm{FNO}}(\mathbf r),
\qquad
D^{\mathrm{ML}}=D^{(0)}+\Delta D_{\mathrm{MACE}}.
$$

MACE produces equivariant atomic features. Radial functions and spherical
harmonics map these features to the potential on a spatial grid. **MAPHO-MACE**
uses this field directly. **MAPHO-FNO** adds a correction that mixes Fourier
components within a physical reciprocal-space cutoff. The backbone and the
prediction heads are trained jointly.

The DFT backend supplies the remaining terms of the operator. A dense
plane-wave Hamiltonian matrix is unnecessary. For semilocal PBE, the predicted
potential and PAW matrices can be used at different k points without
retraining. The potential representation can also be integrated in an atomic
basis. Each backend must still implement its basis and pseudopotential
conventions.

## Getting started

Follow the [Si tutorial](tutorial.md) to prepare PBE data, train a model from
YAML, and calculate a primitive-cell band structure. The tutorial uses the
VASP PAW implementation and requires a private MAPHO-enabled VASP executable.
VASP source, executables, and PAW datasets are not included in this repository.

See [usage.md](usage.md) for installation, command groups, and the independent
orbital-model demonstration that can run without a DFT executable.

## Documentation

| Guide | Content |
|---|---|
| [Tutorial](tutorial.md) | Data preparation, training, and Si bands |
| [Usage](usage.md) | Installation, commands, and development checks |
| [Training configuration](docs/training.md) | YAML settings, data layout, and checkpoints |
| [Package layout](docs/backends.md) | Shared models and backend interfaces |
| [QE fields](docs/field_learning.md) | Potential learning with norm-conserving data |
| [QE backend](docs/qe.md) | Field export and operator reconstruction |
| [Orbital baseline](docs/orbital_baseline.md) | Localized-basis matrices and data conventions |
| [PBE0 extension](docs/vasp_pbe0.md) | Conditional local/PAW learning with reference exchange |
| [Release exclusions](docs/releasing.md) | Separation of public code and private integration files |

## Scope

The current training workflow targets nonmagnetic PBE with fixed PAW or
norm-conserving datasets. The VASP operator interface also supports collinear
spin export and replay, but the tutorial trains a scalar model. PBE0 requires
an additional nonlocal exchange operator. The present hybrid pilot supplies
reference exchange and does not predict the full hybrid Hamiltonian from
structure alone.

The current results concern potentials, PAW matrices, and electronic bands.
Forces and phonons are not provided by the imported-operator workflow.
Pretrained models and benchmark data are not included. Accuracy for a new
material or structural distribution requires a separate evaluation.
