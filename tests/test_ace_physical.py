"""Full diagonalizations with screened/unscreened FFT exchange on a small grid.

The exchange density matrix is fixed. These are controlled numerical tests,
not self-consistent calculations of a material.
"""

import numpy as np
import pytest
import torch

from mapho.ace_operator import ACEOperator
from mapho.exchange import FockExchange


@pytest.fixture(params=[0.106, 0.0], ids=["screened", "unscreened"])
def dense_problem(request):
    rng = torch.Generator().manual_seed(321)
    shape, n, nstate = (5, 5, 5), 125, 6
    occupied = torch.linalg.qr(
        torch.randn(n, 3, dtype=torch.complex128, generator=rng)
    )[0]
    mu = request.param
    options = {"zero_mode": 200.0} if mu == 0 else {}
    fock = FockExchange(
        occupied.T.reshape(1, 3, *shape),
        [[0, 0, 0]],
        [1.0],
        np.eye(3) * 10,
        np.ones((1, *shape), bool),
        mu=mu,
        recompute=False,
        **options,
    )
    identity = torch.eye(n, dtype=torch.complex128)
    exchange = fock.apply(identity.reshape(n, *shape), 0).reshape(n, n).T
    # A nondegenerate one-body spectrum makes convergence of the chosen lowest
    # states measurable. It does not commute with this real-space exchange.
    h0 = torch.diag(torch.arange(n, dtype=torch.float64) * 2).to(torch.complex128)
    full = h0 + exchange
    eigenvalues, eigenvectors = torch.linalg.eigh(full)
    return h0, exchange, identity, eigenvalues, eigenvectors, nstate, rng


def test_full_diagonalization_preserves_construction_eigenstates(dense_problem):
    h0, exchange, identity, exact, states, nstate, _ = dense_problem
    phi = states[:, :nstate]
    ace = ACEOperator.from_samples(phi, exchange @ phi)
    compressed = h0 + ace.apply(identity)
    eigenvalues = torch.linalg.eigvalsh(compressed)
    torch.testing.assert_close(
        eigenvalues[:nstate], exact[:nstate], rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        compressed @ phi, phi * exact[:nstate], rtol=1e-10, atol=1e-10
    )
    # For a negative exchange operator the compression raises, rather than
    # spuriously lowers, eigenvalues outside the represented subspace.
    assert float((eigenvalues - exact).min()) > -1e-10


def test_adaptive_rebuilding_converges_to_full_diagonalization(
    dense_problem, record_property
):
    h0, exchange, identity, exact, _, nstate, rng = dense_problem
    phi = torch.linalg.qr(
        torch.randn(len(exact), nstate, dtype=torch.complex128, generator=rng)
    )[0]
    residual = float("inf")
    for iteration in range(50):
        ace = ACEOperator.from_samples(phi, exchange @ phi)
        eigenvalues, states = torch.linalg.eigh(h0 + ace.apply(identity))
        phi = states[:, :nstate]
        residual = float(
            torch.linalg.vector_norm(
                (h0 + exchange) @ phi - phi * eigenvalues[:nstate], dim=0
            ).max()
        )
        if residual < 1e-9:
            break
    eigenvalue_error = float((eigenvalues[:nstate] - exact[:nstate]).abs().max())
    record_property("ace_rebuilds", iteration + 1)
    record_property("max_eigenvalue_error_eV", eigenvalue_error)
    record_property("max_full_operator_residual_eV", residual)
    assert residual < 1e-9
    assert eigenvalue_error < 1e-10


def test_empty_state_accuracy_requires_enlarging_construction_space(dense_problem):
    h0, exchange, identity, exact, states, nstate, _ = dense_problem
    occupied = states[:, :3]
    small = ACEOperator.from_samples(occupied, exchange @ occupied)
    small_eigenvalues = torch.linalg.eigvalsh(h0 + small.apply(identity))
    torch.testing.assert_close(small_eigenvalues[:3], exact[:3], rtol=1e-11, atol=1e-11)
    assert float((small_eigenvalues[3:nstate] - exact[3:nstate]).abs().max()) > 1e-3
    enlarged = ACEOperator.from_samples(
        states[:, :nstate], exchange @ states[:, :nstate]
    )
    enlarged_eigenvalues = torch.linalg.eigvalsh(h0 + enlarged.apply(identity))
    torch.testing.assert_close(
        enlarged_eigenvalues[:nstate], exact[:nstate], rtol=1e-11, atol=1e-11
    )
