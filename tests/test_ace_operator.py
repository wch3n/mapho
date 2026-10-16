"""Synthetic mathematical checks; no VASP code, data, or binaries are used."""

import pytest
import torch

from mapho.ace_operator import ACEOperator


def problem(dtype=torch.complex128, size=18, states=5):
    rng = torch.Generator().manual_seed(106)
    a = torch.randn(size, size, dtype=dtype, generator=rng)
    exchange = -(a @ a.mH + torch.eye(size, dtype=dtype))
    phi = torch.linalg.qr(torch.randn(size, states, dtype=dtype, generator=rng))[0]
    return exchange, phi


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_exact_action_on_construction_space(dtype):
    exchange, phi = problem(dtype)
    operator = ACEOperator.from_samples(phi, exchange @ phi)
    tolerance = 3e-5 if phi.real.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        operator.apply(phi), exchange @ phi, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        operator.apply(phi[:, 0]), exchange @ phi[:, 0], rtol=tolerance, atol=tolerance
    )
    assert operator.rank == phi.shape[1]
    assert operator.storage_bytes == phi.numel() * phi.element_size()


def test_hermitian_and_negative_semidefinite_on_arbitrary_vectors():
    exchange, phi = problem()
    operator = ACEOperator.from_samples(phi, exchange @ phi)
    matrix = operator.apply(torch.eye(exchange.shape[0], dtype=exchange.dtype))
    torch.testing.assert_close(matrix, matrix.mH, atol=1e-12, rtol=1e-12)
    assert float(torch.linalg.eigvalsh(matrix).max()) < 1e-12


def test_invariant_under_change_of_construction_basis():
    exchange, phi = problem()
    operator = ACEOperator.from_samples(phi, exchange @ phi)
    rng = torch.Generator().manual_seed(63)
    rotation = torch.linalg.qr(torch.randn(5, 5, dtype=phi.dtype, generator=rng))[0]
    # Include nonorthogonal columns and complex phases, not just a permutation.
    transform = rotation @ torch.diag(torch.tensor([1, 2, 3, 4, 5], dtype=phi.dtype))
    changed = ACEOperator.from_samples(phi @ transform, exchange @ phi @ transform)
    torch.testing.assert_close(
        operator.factors @ operator.factors.mH,
        changed.factors @ changed.factors.mH,
        atol=1e-11,
        rtol=1e-12,
    )


def test_full_basis_recovers_entire_operator():
    exchange, phi = problem(states=18)
    operator = ACEOperator.from_samples(phi, exchange @ phi)
    torch.testing.assert_close(
        -operator.factors @ operator.factors.mH, exchange, atol=1e-11, rtol=1e-12
    )


def test_no_claim_of_exactness_outside_construction_space():
    exchange = -torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    basis = torch.eye(3, dtype=torch.float64)
    operator = ACEOperator.from_samples(basis[:, :1], exchange[:, :1])
    assert torch.linalg.vector_norm(operator.apply(basis[:, 1]) - exchange[:, 1]) == 2


def test_hybrid_fraction_is_applied_once():
    exchange, phi = problem()
    full = ACEOperator.from_samples(phi, exchange @ phi)
    quarter = ACEOperator.from_samples(phi, 0.25 * (exchange @ phi))
    torch.testing.assert_close(quarter.apply(phi), 0.25 * full.apply(phi))


@pytest.mark.parametrize(
    "case", ["positive", "singular", "nonhermitian", "nan", "shape", "dtype"]
)
def test_reject_invalid_samples(case):
    phi = torch.eye(3, dtype=torch.float64)
    actions = -phi.clone()
    if case == "positive":
        actions = phi.clone()
    elif case == "singular":
        actions[:, 0] = 0
    elif case == "nonhermitian":
        actions[0, 1] = 0.2
    elif case == "nan":
        actions[0, 0] = float("nan")
    elif case == "shape":
        actions = actions[:, :2]
    elif case == "dtype":
        actions = actions.float()
    with pytest.raises(ValueError):
        ACEOperator.from_samples(phi, actions)


@pytest.mark.parametrize("rtol", [0, -1, 1, float("nan"), float("inf")])
def test_reject_invalid_tolerance(rtol):
    exchange, phi = problem()
    with pytest.raises(ValueError, match="rtol"):
        ACEOperator.from_samples(phi, exchange @ phi, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_cpu():
    exchange, phi = problem()
    cpu = ACEOperator.from_samples(phi, exchange @ phi)
    gpu = ACEOperator.from_samples(phi.cuda(), (exchange @ phi).cuda())
    torch.testing.assert_close(
        gpu.apply(phi.cuda()).cpu(), cpu.apply(phi), atol=1e-11, rtol=1e-12
    )
