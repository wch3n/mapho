"""Analytic synthetic labels for plumbing checks, not a DFT surrogate dataset."""

import math
import numpy as np
import torch
from ase import Atoms
from e3nn import o3

from .data import Sample
from .graph import make_graph
from .model import orbital_irreps


def synthetic_sample(seed, basis="sp", cutoff=4.0):
    rng = np.random.default_rng(seed)
    cell = np.diag(rng.uniform(4.0, 4.7, 3))
    cell[1, 0], cell[2, 1] = rng.uniform(-0.3, 0.3, 2)
    fractional = np.array([[0.12, 0.13, 0.11], [0.46, 0.40, 0.36]])
    fractional += rng.normal(0, 0.025, fractional.shape)
    atoms = Atoms(numbers=[6, 8], positions=fractional @ cell, cell=cell, pbc=True)
    graph = make_graph(atoms, cutoff)
    orbitals = orbital_irreps(basis)
    vectors = graph.vectors()
    distance = vectors.norm(dim=-1)
    onsite = graph.onsite
    i, j = graph.pairs[:, 3:].T
    y = torch.cat(
        [
            o3.spherical_harmonics(
                ir.l, vectors, normalize=True, normalization="component"
            ).repeat(1, mul)
            for mul, ir in orbitals
        ],
        dim=-1,
    )
    parity = torch.tensor([ir.p for mul, ir in orbitals for _ in range(mul * ir.dim)])
    envelope = 0.5 * (1 + torch.cos(math.pi * distance / cutoff))
    hopping = -0.45 * torch.exp(-0.65 * distance) * envelope
    species = 1 + 0.025 * (graph.numbers[i] + graph.numbers[j]).float()
    blocks = (
        (hopping * species)[:, None, None] * y[:, :, None] * (y * parity)[:, None, :]
    )
    blocks[onsite] = 0
    # Environment-dependent onsite crystal fields make this a many-body task.
    crystal = torch.zeros(len(graph.numbers), orbitals.dim, orbitals.dim)
    weight = 0.05 * torch.exp(-distance) * envelope * (~onsite)
    crystal.index_add_(0, i, weight[:, None, None] * y[:, :, None] * y[:, None, :])
    levels = torch.tensor(
        [0.5 * ir.l for mul, ir in orbitals for _ in range(mul * ir.dim)]
    )
    node = graph.pairs[onsite, 3]
    diagonal = levels[None] - 0.12 * graph.numbers[node, None]
    blocks[onsite] = torch.diag_embed(diagonal) + crystal[node]
    blocks = 0.5 * (blocks + blocks[graph.reverse].mT)
    return Sample(
        graph,
        blocks,
        basis,
        {
            "source": "analytic_synthetic",
            "seed": seed,
            "energy_reference": "fixed synthetic onsite reference",
            "orthogonal_basis": True,
        },
    ).validate()
