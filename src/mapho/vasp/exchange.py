"""Construct MAPHO ACE from full exchange actions in a MAPHO-defined format.

The private VASP hook supplies the nonlocal remainder matching its exported
PAW matrices. Its output is a Hermitian coefficient-space action in eV,
including the hybrid fraction, rather than S^-1 H acting on coefficients.
This Python module contains no VASP source or source-patching anchors.
"""

import json
from pathlib import Path

import numpy as np
import torch

from ..ace_operator import ACEOperator
from ..file_utils import sha
from .ace import apply_ace, read_ace, read_exchange_samples, write_ace


def build_ace_from_samples(samples, output):
    """PAW-normalized columns are allowed; Euclidean orthonormality is not assumed."""
    archive = read_exchange_samples(samples)
    factors, errors = [], []
    for orbitals, actions in zip(archive["orbitals"], archive["actions"]):
        phi, w = torch.from_numpy(orbitals), torch.from_numpy(actions)
        operator = ACEOperator.from_samples(phi, w)
        factors.append(operator.factors.detach().resolve_conj().numpy())
        errors.append(
            float(torch.linalg.vector_norm(operator.apply(phi) - w, dim=0).max())
        )
    write_ace(output, dict(archive, factors=factors))
    return dict(
        schema="mapho.exchange_factorization.v1",
        backend="mapho.ace_operator.ACEOperator",
        samples=str(Path(samples).resolve()),
        samples_sha256=sha(samples),
        ace_file=str(Path(output).resolve()),
        ace_sha256=sha(output),
        action_max_error_eV=errors,
        native_ace_used_for_factorization=False,
    )


def compare_native_factors(samples, mapho_factors, native_factors):
    """Compare actions, since different Cholesky conventions can rotate factors."""
    samples = read_exchange_samples(samples)
    ours, native = read_ace(mapho_factors), read_ace(native_factors)
    for other in (ours, native):
        for key in ("dims", "grid", "kpoints", "cell_A"):
            if not np.allclose(other[key], samples[key], rtol=0, atol=1e-10):
                raise ValueError("Exchange comparison metadata mismatch")
        if abs(other["fraction"] - samples["fraction"]) > 1e-12:
            raise ValueError("Exchange fraction mismatch")
        if any(
            not np.array_equal(a, b)
            for a, b in zip(other["indices"], samples["indices"])
        ):
            raise ValueError("Exchange reciprocal ordering mismatch")
    rows = []
    rng = np.random.default_rng(512)
    for k, (phi, target) in enumerate(zip(samples["orbitals"], samples["actions"])):
        probes = rng.normal(size=(len(phi), 3)) + 1j * rng.normal(size=(len(phi), 3))
        probes /= np.linalg.norm(probes, axis=0)
        ours_actions = apply_ace(ours, k, phi)
        rows.append(
            dict(
                k_index=k,
                construction_action_max_error_eV=float(
                    np.linalg.norm(ours_actions - target, axis=0).max()
                ),
                native_construction_max_difference_eV=float(
                    np.linalg.norm(
                        ours_actions - apply_ace(native, k, phi), axis=0
                    ).max()
                ),
                native_probe_max_difference_eV=float(
                    np.linalg.norm(
                        apply_ace(ours, k, probes) - apply_ace(native, k, probes),
                        axis=0,
                    ).max()
                ),
            )
        )
    passed = all(
        value < 1e-8
        for row in rows
        for key, value in row.items()
        if key.endswith("_eV")
    )
    return dict(
        rows=rows,
        passed=passed,
        comparison="Actions only; factors need not be identical",
    )


def main(argv=None):
    """Compatibility entry point; argument parsing lives in mapho.commands.vasp."""
    from ..commands.vasp.exchange import main as command

    return command(argv)


if __name__ == "__main__":
    main()
