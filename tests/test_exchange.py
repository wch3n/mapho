import numpy as np
import pytest
import torch

from mapho.exchange import (
    ScreenedExchange,
    FockExchange,
    screened_kernel,
    coulomb_kernel,
    gygi_baldereschi_zero,
    pw_to_grid,
    pw_mask,
)


def fixture_operator():
    rng = np.random.default_rng(51)
    shape = (5, 5, 5)
    orbitals = []
    for _ in range(2):
        z = rng.normal(size=(125, 2)) + 1j * rng.normal(size=(125, 2))
        orbitals.append(np.linalg.qr(z)[0].T.reshape(2, *shape))
    phi = torch.tensor(np.stack(orbitals), dtype=torch.complex128)
    k = np.array([[0, 0, 0], [0.25, -0.25, 0.1]])
    weights = np.array([0.4, 0.6])
    cell = np.array([[8, 0, 0], [1, 7, 0], [0.5, 0.2, 9.0]])
    masks = np.ones((2, *shape), bool)
    return ScreenedExchange(phi, k, weights, cell, masks), rng


def test_exchange_linearity_hermiticity_and_sign():
    op, rng = fixture_operator()
    z = torch.tensor(
        rng.normal(size=(2, *op.shape)) + 1j * rng.normal(size=(2, *op.shape))
    )
    g = torch.tensor(np.exp(0.1 * rng.normal(size=op.shape)))
    a, b = 1.2 + 0.7j, -0.2 + 0.3j
    y = op.apply(z, 1, g)
    torch.testing.assert_close(
        op.apply((a * z[0] + b * z[1])[None], 1, g)[0], a * y[0] + b * y[1]
    )
    torch.testing.assert_close(
        torch.vdot(z[0].flatten(), y[1].flatten()),
        torch.vdot(y[0].flatten(), z[1].flatten()),
    )
    assert (z.conj() * y).sum((-3, -2, -1)).real.max() < 0


def test_exchange_matches_explicit_two_position_kernel():
    op, rng = fixture_operator()
    n = np.prod(op.shape)
    # Build the convolution matrix by applying its known spectral eigenvalues
    # to coordinate impulses, then multiply by the occupied density matrix.
    impulses = torch.eye(n, dtype=torch.complex128).reshape(n, *op.shape)
    direct = torch.zeros((n, n), dtype=torch.complex128)
    for phi, kernel in zip(op.orbitals, op.kernels[0]):
        c = (
            torch.fft.ifftn(
                torch.fft.fftn(impulses, dim=(-3, -2, -1)) * kernel, dim=(-3, -2, -1)
            )
            .reshape(n, n)
            .T
        )
        p = phi.reshape(2, n).T
        direct -= (p @ p.conj().T) * c
    z = torch.tensor(
        rng.normal(size=(3, *op.shape)) + 1j * rng.normal(size=(3, *op.shape))
    )
    torch.testing.assert_close(op.apply(z, 0).reshape(3, n), z.reshape(3, n) @ direct.T)


def test_occupied_unitary_invariance():
    op, rng = fixture_operator()
    z = torch.tensor(
        rng.normal(size=(1, *op.shape)) + 1j * rng.normal(size=(1, *op.shape))
    )
    before = op.apply(z, 0)
    u = torch.tensor(
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    )
    op.orbitals = torch.einsum("ab,qbijk->qaijk", u, op.orbitals)
    torch.testing.assert_close(op.apply(z, 0), before)


def test_gate_gradient_and_periodic_translation():
    op, rng = fixture_operator()
    z = torch.tensor(
        rng.normal(size=(1, *op.shape)) + 1j * rng.normal(size=(1, *op.shape))
    )
    g = torch.ones(op.shape, dtype=torch.float64, requires_grad=True)
    y = op.apply(z, 1, g)
    loss = y.abs().square().sum()
    loss.backward()
    delta = torch.zeros_like(g)
    delta[1, 2, 3] = 1e-5
    with torch.no_grad():
        finite = (
            op.apply(z, 1, g + delta).abs().square().sum()
            - op.apply(z, 1, g - delta).abs().square().sum()
        ) / 2e-5
    torch.testing.assert_close(g.grad[1, 2, 3], finite, rtol=1e-7, atol=1e-7)
    shifts = (1, -1, 2)
    dims = (-3, -2, -1)
    op.orbitals = op.orbitals.roll(shifts, dims)
    torch.testing.assert_close(
        op.apply(z.roll(shifts, dims), 1, g.roll(shifts, dims)), y.roll(shifts, dims)
    )


def test_pw_norm_projection_and_alias_rejection():
    mill = np.array([[0, 0, 0], [1, 0, 0], [-1, 1, 0]])
    c = np.array([[1, 2j, 3.0]])
    grid = pw_to_grid(c, mill, (5, 5, 5))
    assert np.isclose(np.sum(abs(grid) ** 2), np.sum(abs(c) ** 2))
    assert pw_mask(mill, (5, 5, 5)).sum() == 3
    with pytest.raises(ValueError, match="aliases"):
        pw_to_grid(c, np.array([[0, 0, 0], [5, 0, 0], [1, 0, 0]]), (5, 5, 5))
    w = screened_kernel((5, 5, 5), np.eye(3) * 8, np.zeros(3), mu=0.106)
    assert np.isclose(w[0, 0, 0], np.pi / 0.106**2)


def test_unscreened_singular_correction_scaling_and_required_zero_mode():
    cell = np.array([[0, 4.0, 4.0], [4.0, 0, 4.0], [4.0, 4.0, 0]])
    zero = gygi_baldereschi_zero(cell, 100, 400, [2, 2, 2])
    # At fixed dimensionless resolution, a length dilation scales w(G) as L^2.
    scale = 1.3
    scaled = gygi_baldereschi_zero(
        scale * cell, 100 / scale**2, 400 / scale**2, [2, 2, 2]
    )
    assert np.isclose(scaled, zero * scale**2, rtol=1e-11)
    w = coulomb_kernel((5, 5, 5), cell, [0, 0, 0], zero)
    assert w[0, 0, 0] == zero
    g = 2 * np.pi * np.linalg.inv(cell) @ np.array([1, 0, 0])
    assert np.isclose(w[1, 0, 0], 4 * np.pi / np.dot(g, g))
    op, _ = fixture_operator()
    with pytest.raises(ValueError, match="zero mode"):
        FockExchange(
            op.orbitals, op.kpoints, [0.4, 0.6], np.eye(3) * 8, op.masks.numpy(), mu=0
        )


def test_unscreened_zero_mode_adds_the_occupied_projector():
    rng = np.random.default_rng(92)
    shape = (5, 5, 5)
    n = np.prod(shape)
    cell = np.eye(3) * 8
    phi = np.linalg.qr(rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3)))[0].T
    phi = torch.tensor(phi.reshape(1, 3, *shape))
    masks = np.ones((1, *shape), bool)
    kwargs = dict(mu=0, alpha=0.25, rydberg_eV=13.605693122994)
    first = FockExchange(phi, [[0, 0, 0]], [1.0], cell, masks, zero_mode=10, **kwargs)
    second = FockExchange(phi, [[0, 0, 0]], [1.0], cell, masks, zero_mode=17, **kwargs)
    psi = torch.tensor(rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape)))
    gate = torch.tensor(np.exp(0.1 * rng.normal(size=shape)))
    # Independent analytic form of the zero-momentum contribution, including
    # multiplication on both sides and the same-spin/volume normalization.
    orbitals = (phi[0] * gate).flatten(1)
    projection = (psi.flatten(1) @ orbitals.conj().T) @ orbitals
    expected = -2 * 0.25 * kwargs["rydberg_eV"] * 7 / np.linalg.det(cell) * projection
    torch.testing.assert_close(
        (second.apply(psi, 0, gate) - first.apply(psi, 0, gate)).flatten(1), expected
    )
    out = second.apply(psi, 0, gate)
    torch.testing.assert_close(
        torch.vdot(psi[0].flatten(), out[1].flatten()),
        torch.vdot(out[0].flatten(), psi[1].flatten()),
    )
    assert (psi.conj() * out).sum((-3, -2, -1)).real.max() < 0
