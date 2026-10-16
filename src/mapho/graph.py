"""Periodic directed graphs, including every lattice image inside the cutoff."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list


@dataclass
class Graph:
    numbers: torch.Tensor
    positions: torch.Tensor
    cells: torch.Tensor
    pbc: torch.Tensor
    batch: torch.Tensor
    pairs: torch.Tensor  # [Rx, Ry, Rz, i, j]; zero-based atom indices
    reverse: torch.Tensor
    cutoff: float

    @property
    def onsite(self):
        return (self.pairs[:, :3] == 0).all(-1) & (self.pairs[:, 3] == self.pairs[:, 4])

    def vectors(self):
        i, j = self.pairs[:, 3:].T
        shift = torch.einsum(
            "ei,eij->ej",
            self.pairs[:, :3].to(self.positions),
            self.cells[self.batch[i]],
        )
        return self.positions[j] - self.positions[i] + shift

    def to(self, device="cpu", dtype=None):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(
                    device=device, dtype=dtype if value.is_floating_point() else None
                )
            values[field.name] = value
        return Graph(**values)


def reverse_indices(pairs):
    pairs = np.asarray(pairs)
    keys = [tuple(map(int, row)) for row in pairs]
    lookup = {key: i for i, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError("Duplicate atom-pair blocks")
    reverse = []
    for x, y, z, i, j in keys:
        key = (-x, -y, -z, j, i)
        if key not in lookup:
            raise ValueError(f"Missing reverse block {key}")
        reverse.append(lookup[key])
    return np.asarray(reverse, dtype=np.int64)


def make_graph(atoms: Atoms, cutoff: float, dtype=torch.float32):
    if not np.isfinite(cutoff) or cutoff <= 0 or len(atoms) == 0:
        raise ValueError("A positive finite cutoff and at least one atom are required")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("Geometry must be finite")
    if np.any(atoms.numbers < 1) or np.any(atoms.numbers > 118):
        raise ValueError("Atomic numbers must be in [1, 118]")
    if atoms.pbc.any() and np.linalg.matrix_rank(atoms.cell.array) != 3:
        raise ValueError(
            "Periodic structures require a full-rank cell, including vacuum vectors"
        )
    i, j, shifts = neighbor_list("ijS", atoms, cutoff, self_interaction=True)
    pairs = np.column_stack((shifts, i, j)).astype(np.int64)
    order = np.lexsort(tuple(pairs[:, k] for k in reversed(range(5))))
    pairs = pairs[order]
    graph = Graph(
        torch.as_tensor(atoms.numbers.copy(), dtype=torch.long),
        torch.as_tensor(atoms.positions.copy(), dtype=dtype),
        torch.as_tensor(atoms.cell.array.copy()[None], dtype=dtype),
        torch.as_tensor(atoms.pbc.copy()[None], dtype=torch.bool),
        torch.zeros(len(atoms), dtype=torch.long),
        torch.as_tensor(pairs, dtype=torch.long),
        torch.as_tensor(reverse_indices(pairs), dtype=torch.long),
        float(cutoff),
    )
    if (graph.vectors().norm(dim=-1)[~graph.onsite] < 1e-7).any():
        raise ValueError("Distinct atoms or images coincide")
    return graph


def batch_graphs(graphs):
    """Disjoint union; periodic cells and reverse edges remain independent."""
    if not graphs or any(g.cells.shape[0] != 1 for g in graphs):
        raise ValueError("Expected a nonempty list of individual graphs")
    if any(g.cutoff != graphs[0].cutoff for g in graphs):
        raise ValueError("All graphs must use the same cutoff")
    pairs, reverse, batches = [], [], []
    node_offset = edge_offset = 0
    for b, g in enumerate(graphs):
        pair = g.pairs.clone()
        pair[:, 3:] += node_offset
        pairs.append(pair)
        reverse.append(g.reverse + edge_offset)
        batches.append(torch.full_like(g.batch, b))
        node_offset += len(g.numbers)
        edge_offset += len(g.pairs)
    return Graph(
        torch.cat([g.numbers for g in graphs]),
        torch.cat([g.positions for g in graphs]),
        torch.cat([g.cells for g in graphs]),
        torch.cat([g.pbc for g in graphs]),
        torch.cat(batches),
        torch.cat(pairs),
        torch.cat(reverse),
        graphs[0].cutoff,
    )
