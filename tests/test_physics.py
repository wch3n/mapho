from dataclasses import replace

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mapho.bands import band_energies, bloch_matrix
from mapho.graph import make_graph, batch_graphs, reverse_indices
from mapho.model import HamiltonianModel, ModelConfig
from mapho.synthetic import synthetic_sample


@pytest.mark.parametrize("basis", ["s", "sp", "spd"])
def test_rotation_inversion_translation_hermiticity(basis):
    model = HamiltonianModel(ModelConfig(basis=basis, channels=4, layers=2)).double()
    graph = synthetic_sample(10, basis).graph.to(dtype=torch.float64)
    h = model(graph)
    torch.testing.assert_close(h, h[graph.reverse].mT, atol=0, rtol=0)
    for determinant in (1, -1):
        q = o3.rand_matrix(dtype=torch.float64) * determinant
        rotated = replace(
            graph, positions=graph.positions @ q.T + 1.234, cells=graph.cells @ q.T
        )
        d = model.orbitals.D_from_matrix(q)
        torch.testing.assert_close(model(rotated), d @ h @ d.T, atol=2e-6, rtol=2e-5)
        k = [[0, 0, 0], [0.17, 0.23, -0.08]]
        torch.testing.assert_close(
            band_energies(model(rotated), rotated, k),
            band_energies(h, graph, k),
            atol=3e-6,
            rtol=2e-5,
        )


def test_permutation_and_batch_independence():
    model = HamiltonianModel(ModelConfig(channels=4, layers=1))
    graph = synthetic_sample(12).graph
    other = synthetic_sample(13).graph
    h = model(graph)
    swapped = Atoms(
        numbers=graph.numbers.flip(0).numpy(),
        positions=graph.positions.flip(0).numpy(),
        cell=graph.cells[0].numpy(),
        pbc=True,
    )
    reordered = make_graph(swapped, graph.cutoff)
    reference = {tuple(row): block for row, block in zip(graph.pairs.tolist(), h)}
    expected = torch.stack(
        [
            reference[tuple(row[:3] + [1 - row[3], 1 - row[4]])]
            for row in reordered.pairs.tolist()
        ]
    )
    torch.testing.assert_close(model(reordered), expected, atol=1e-6, rtol=1e-5)
    joined = model(batch_graphs([graph, other]))
    torch.testing.assert_close(joined[: len(h)], h, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(joined[len(h) :], model(other), atol=1e-6, rtol=1e-5)


def test_multiple_periodic_images_and_triclinic_graph():
    atoms = Atoms(
        "C",
        positions=[[0, 0, 0]],
        cell=[[1, 0, 0], [0.4, 1.3, 0], [0.2, 0.3, 1.5]],
        pbc=True,
    )
    graph = make_graph(atoms, 2.1, dtype=torch.float64)
    expected = set()
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                if np.linalg.norm(np.array([x, y, z]) @ atoms.cell.array) < 2.1:
                    expected.add((x, y, z, 0, 0))
    assert set(map(tuple, graph.pairs.tolist())) == expected
    torch.testing.assert_close(graph.vectors()[graph.reverse], -graph.vectors())
    assert sum(graph.onsite).item() == 1


def test_chain_bands_and_nonorthogonal_overlap():
    graph = make_graph(
        Atoms("C", positions=[[0, 0, 0]], cell=[1, 5, 5], pbc=[True, False, False]),
        1.1,
        dtype=torch.float64,
    )
    h = torch.full((3, 1, 1), -0.7, dtype=torch.float64)
    h[graph.onsite] = 0.4
    s = torch.full_like(h, 0.05)
    s[graph.onsite] = 1.0
    k = torch.tensor([[0, 0, 0], [0.19, 0, 0], [0.5, 0, 0]], dtype=torch.float64)
    c = torch.cos(2 * torch.pi * k[:, 0])
    torch.testing.assert_close(band_energies(h, graph, k)[:, 0], 0.4 - 1.4 * c)
    torch.testing.assert_close(
        band_energies(h, graph, k, overlap=s)[:, 0], (0.4 - 1.4 * c) / (1 + 0.1 * c)
    )
    with pytest.raises(ValueError, match="positive definite"):
        band_energies(h, graph, k, overlap=-s)
    h[0] += 0.1
    with pytest.raises(ValueError, match="Hermitian"):
        band_energies(h, graph, k)


def test_complex_generalized_eigenproblem():
    graph = make_graph(Atoms("C", positions=[[0, 0, 0]]), 1.0, dtype=torch.float64)
    h = torch.tensor([[[1, 0.3 + 0.4j], [0.3 - 0.4j, 2]]], dtype=torch.complex128)
    s = torch.tensor([[[1.2, 0.1j], [-0.1j, 0.9]]], dtype=torch.complex128)
    from scipy.linalg import eigh

    expected = torch.tensor(eigh(h[0].numpy(), s[0].numpy(), eigvals_only=True))
    torch.testing.assert_close(band_energies(h, graph, [[0, 0, 0]], s)[0], expected)


def test_isolated_atom_and_finite_gradients():
    model = HamiltonianModel(ModelConfig(basis="spd", channels=2, layers=1))
    isolated = make_graph(Atoms("C", positions=[[0, 0, 0]]), 4.0)
    assert torch.isfinite(model(isolated)).all()
    sample = synthetic_sample(10, "spd")
    loss = (model(sample.graph) - sample.hamiltonian).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


def test_reverse_validation():
    with pytest.raises(ValueError, match="Missing reverse"):
        reverse_indices([[1, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="Duplicate"):
        reverse_indices([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


def test_orbital_coupling_is_orthonormal():
    model = HamiltonianModel(ModelConfig(basis="spd", channels=2, layers=1))
    transform = model.matrix_basis.flatten(0, 1)
    torch.testing.assert_close(
        transform.T @ transform, torch.eye(81), atol=3e-7, rtol=3e-7
    )
