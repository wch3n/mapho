"""Versioned, code-independent training samples in an explicit orbital gauge."""

from dataclasses import dataclass, field
import json
from pathlib import Path

from ase import Atoms
import numpy as np
import torch

from .graph import Graph, make_graph
from .model import orbital_irreps


@dataclass
class Sample:
    graph: Graph
    hamiltonian: torch.Tensor
    basis: str
    metadata: dict = field(default_factory=dict)

    def validate(self):
        n = orbital_irreps(self.basis).dim
        if len(self.graph.cells) != 1:
            raise ValueError("A sample must contain one structure")
        if self.hamiltonian.shape != (len(self.graph.pairs), n, n):
            raise ValueError(
                "Hamiltonian dimensions do not match graph and orbital basis"
            )
        if not torch.isfinite(self.hamiltonian).all():
            raise ValueError("Hamiltonian contains nonfinite values")
        reverse = self.graph.reverse.to(self.hamiltonian.device)
        if not torch.allclose(
            self.hamiltonian, self.hamiltonian[reverse].mH, atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Hamiltonian violates reverse-edge Hermiticity")
        return self


def save_sample(path, sample):
    sample.validate()
    g = sample.graph.to("cpu")
    metadata = dict(sample.metadata)
    metadata.update(
        schema_version=1,
        basis=str(orbital_irreps(sample.basis)),
        basis_convention="e3nn_real_spherical_harmonics",
        spinful=False,
        length_unit="angstrom",
        energy_unit="eV",
        cutoff=g.cutoff,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting training labels accidentally.
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            numbers=g.numbers.numpy(),
            positions=g.positions.numpy(),
            cell=g.cells[0].numpy(),
            pbc=g.pbc[0].numpy(),
            pairs=g.pairs.numpy(),
            hamiltonian=sample.hamiltonian.detach().cpu().numpy(),
            metadata=np.asarray(json.dumps(metadata)),
        )


def load_sample(path, dtype=torch.float32):
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"]))
        required = {
            "schema_version": 1,
            "basis_convention": "e3nn_real_spherical_harmonics",
            "length_unit": "angstrom",
            "energy_unit": "eV",
            "spinful": False,
        }
        for key, value in required.items():
            if metadata.get(key) != value:
                raise ValueError(f"Unsupported or missing {key}: {metadata.get(key)}")
        atoms = Atoms(
            numbers=source["numbers"],
            positions=source["positions"],
            cell=source["cell"],
            pbc=source["pbc"],
        )
        graph = make_graph(atoms, metadata["cutoff"], dtype=dtype)
        if not np.array_equal(graph.pairs.numpy(), source["pairs"]):
            raise ValueError("Stored graph is not the complete canonical cutoff graph")
        h = source["hamiltonian"].copy()
    # Preserve imaginary parts for inspection; the real-valued learner rejects them.
    target_dtype = (
        (torch.complex128 if dtype == torch.float64 else torch.complex64)
        if np.iscomplexobj(h)
        else dtype
    )
    sample = Sample(
        graph, torch.as_tensor(h, dtype=target_dtype), metadata["basis"], metadata
    )
    return sample.validate()
