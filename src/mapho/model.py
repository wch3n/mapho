"""O(3)-equivariant message passing and coupled orbital-matrix readout."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from ._e3nn import o3, NormActivation

from .graph import Graph


BASIS = {"s": "0e", "sp": "0e + 1o", "spd": "0e + 1o + 2e"}


def orbital_irreps(basis):
    irreps = o3.Irreps(BASIS.get(basis, basis))
    if not irreps or any(ir.l > 2 or ir.p != (-1) ** ir.l for _, ir in irreps):
        raise ValueError(
            "Use complete s, p, or d shells with physical inversion parity"
        )
    return irreps


@dataclass
class ModelConfig:
    basis: str = "sp"
    cutoff: float = 4.0
    channels: int = 8
    layers: int = 2
    radial_size: int = 12
    radial_hidden: int = 32
    neighbor_scale: float = 4.0

    def __post_init__(self):
        orbital_irreps(self.basis)
        if (
            self.channels < 2
            or self.layers < 1
            or self.radial_size < 2
            or self.radial_hidden < 1
        ):
            raise ValueError("Invalid model dimensions")
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not math.isfinite(self.neighbor_scale) or self.neighbor_scale <= 0:
            raise ValueError("neighbor_scale must be finite and positive")

    def as_dict(self):
        return asdict(self)


class RadialTensorProduct(nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, config):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out,
            shared_weights=False,
            internal_weights=False,
        )
        self.radial = nn.Sequential(
            nn.Linear(config.radial_size, config.radial_hidden),
            nn.SiLU(),
            nn.Linear(config.radial_hidden, self.tp.weight_numel),
        )

    def forward(self, x, sh, radial, envelope):
        return self.tp(x, sh, self.radial(radial)) * envelope[:, None]


class HamiltonianModel(nn.Module):
    """Real, spinless Hamiltonian in a common atom-centred orbital basis.

    For an orthogonal transformation Q, each predicted block transforms as
    H(Qr) = D(Q) H(r) D(Q)^T. Reverse-edge averaging enforces
    H_ij(R) = H_ji(-R)^T without altering this transformation law.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.orbitals = orbital_irreps(c.basis)
        lmax = 2 * self.orbitals.lmax
        # Both parities permit environmental pseudoscalars and axial tensors.
        hidden = [(c.channels, (0, 1)), (c.channels // 2, (0, -1))]
        hidden += [
            (c.channels // 2, (l, p)) for l in range(1, lmax + 1) for p in (-1, 1)
        ]
        self.hidden = o3.Irreps(hidden)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.embedding = nn.Embedding(119, c.channels)
        self.messages = nn.ModuleList(
            [
                RadialTensorProduct(self.hidden, self.sh_irreps, self.hidden, c)
                for _ in range(c.layers)
            ]
        )
        self.self_connections = nn.ModuleList(
            [o3.Linear(self.hidden, self.hidden, biases=True) for _ in range(c.layers)]
        )
        self.activations = nn.ModuleList(
            [
                NormActivation(
                    self.hidden, torch.nn.functional.silu, normalize=True, epsilon=1e-8
                )
                for _ in range(c.layers)
            ]
        )
        coupling = o3.FullTensorProduct(self.orbitals, self.orbitals)
        # This orthonormal change of basis maps coupled coefficients to H_ab.
        with torch.no_grad():
            eye = torch.eye(self.orbitals.dim)
            matrix_basis = coupling(eye[:, None, :], eye[None, :, :])
        self.register_buffer("matrix_basis", matrix_basis)
        self.edge_readout = RadialTensorProduct(
            self.hidden + self.hidden, self.sh_irreps, coupling.irreps_out, c
        )
        self.onsite_readout = o3.Linear(self.hidden, coupling.irreps_out, biases=True)
        self.register_buffer(
            "radial_centers", torch.linspace(0, c.cutoff, c.radial_size)
        )

    def forward(self, graph: Graph):
        if abs(graph.cutoff - self.config.cutoff) > 1e-7:
            raise ValueError("Graph and model cutoffs differ")
        onsite = graph.onsite
        edge = graph.pairs[~onsite]
        i, j = edge[:, 3:].T
        vectors = graph.vectors()[~onsite]
        distances = vectors.norm(dim=-1)
        sh = o3.spherical_harmonics(
            self.sh_irreps, vectors, normalize=True, normalization="component"
        )
        width = self.config.cutoff / (self.config.radial_size - 1)
        radial = torch.exp(-(((distances[:, None] - self.radial_centers) / width) ** 2))
        envelope = 0.5 * (1 + torch.cos(math.pi * distances / self.config.cutoff))
        envelope = envelope * (distances < self.config.cutoff)
        x = torch.nn.functional.pad(
            self.embedding(graph.numbers), (0, self.hidden.dim - self.config.channels)
        )
        for message, connection, activation in zip(
            self.messages, self.self_connections, self.activations
        ):
            aggregate = torch.zeros_like(x)
            if len(edge):
                aggregate.index_add_(0, i, message(x[j], sh, radial, envelope))
            x = activation(
                connection(x) + aggregate / math.sqrt(self.config.neighbor_scale)
            )
        coefficients = x.new_zeros((len(graph.pairs), self.matrix_basis.shape[-1]))
        coefficients[onsite] = self.onsite_readout(x[graph.pairs[onsite, 3]])
        if len(edge):
            coefficients[~onsite] = self.edge_readout(
                torch.cat((x[i], x[j]), -1), sh, radial, envelope
            )
        blocks = torch.einsum("ek,abk->eab", coefficients, self.matrix_basis)
        return 0.5 * (blocks + blocks[graph.reverse].transpose(-1, -2))
