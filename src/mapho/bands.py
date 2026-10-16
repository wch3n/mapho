"""Cell-periodic Bloch sums, with optional nonorthogonal-basis diagonalization."""

import math
import torch


def bloch_matrix(blocks, graph, kpoints):
    """H(k) = sum_R H_ij(R) exp(2 pi i k.R), with fractional reciprocal k.

    Atom-position phases are absent in this cell-periodic gauge. The input
    graph must describe one cell, not a training batch.
    """
    if len(graph.cells) != 1:
        raise ValueError("Bloch assembly requires one structure")
    if (
        blocks.ndim != 3
        or blocks.shape[0] != len(graph.pairs)
        or blocks.shape[1] != blocks.shape[2]
    ):
        raise ValueError("Expected one square orbital block per graph edge")
    real_dtype = blocks.real.dtype
    k = torch.as_tensor(kpoints, dtype=real_dtype, device=blocks.device)
    if k.ndim != 2 or k.shape[1] != 3 or not torch.isfinite(k).all():
        raise ValueError("kpoints must have finite shape (Nk, 3)")
    if not torch.isfinite(blocks).all():
        raise ValueError("Matrix blocks must be finite")
    pairs = graph.pairs.to(blocks.device)
    phase = torch.exp(2j * math.pi * (k @ pairs[:, :3].to(k).T))
    norb = blocks.shape[-1]
    dim = len(graph.numbers) * norb
    orbital = torch.arange(norb, device=blocks.device)
    rows = pairs[:, 3, None] * norb + orbital
    cols = pairs[:, 4, None] * norb + orbital
    indices = (rows[:, :, None] * dim + cols[:, None, :]).reshape(-1)
    values = (phase[:, :, None, None] * blocks[None]).reshape(len(k), -1)
    result = torch.zeros((len(k), dim * dim), dtype=values.dtype, device=values.device)
    result.index_add_(1, indices, values)
    return result.reshape(len(k), dim, dim)


def _check_hermitian(matrix, name):
    if not torch.allclose(matrix, matrix.mH, atol=1e-5, rtol=1e-5):
        raise ValueError(
            f"{name} is not Hermitian; check reverse blocks and lattice shifts"
        )


def band_energies(blocks, graph, kpoints, overlap=None):
    """Solve H c = E S c. S defaults to identity and is never learned here."""
    hk = bloch_matrix(blocks, graph, kpoints)
    _check_hermitian(hk, "Hamiltonian")
    if overlap is not None:
        sk = bloch_matrix(
            overlap.to(device=blocks.device, dtype=hk.dtype), graph, kpoints
        )
        _check_hermitian(sk, "Overlap")
        try:
            lower = torch.linalg.cholesky(sk)
        except torch.linalg.LinAlgError as error:
            raise ValueError(
                "Overlap must be positive definite at every k point"
            ) from error
        left = torch.linalg.solve_triangular(lower, hk, upper=False)
        hk = torch.linalg.solve_triangular(lower, left.mH, upper=False).mH
    return torch.linalg.eigvalsh(0.5 * (hk + hk.mH))
