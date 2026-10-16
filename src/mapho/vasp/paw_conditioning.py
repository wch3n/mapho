"""Equivariant PAW readouts in coordinates scaled by augmentation overlap.

Q is the PAW augmentation matrix, not the full positive overlap operator S.
The invertible |Q| scaling below is a conditioning choice, not an exact
Hamiltonian-error norm or a proof excluding ghost states. Native spectra must
still be validated. Physical matrices are reconstructed in float64.
"""

import numpy as np
import torch
from torch import nn
from .model import SpeciesPAWHead, PAWModel


def scalar_projection(matrix, channels):
    """O(3)-invariant part, retaining every same-l radial-channel coupling."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n = sum(2 * l + 1 for l in channels)
    if matrix.shape != (n, n):
        raise ValueError("Projector dimensions differ")
    result = np.zeros_like(matrix)
    starts = np.cumsum([0] + [2 * l + 1 for l in channels])
    for a, l in enumerate(channels):
        sa = slice(starts[a], starts[a + 1])
        for b, k in enumerate(channels):
            if l == k:
                sb = slice(starts[b], starts[b + 1])
                result[sa, sb] = (
                    np.eye(2 * l + 1) * np.trace(matrix[sa, sb]) / (2 * l + 1)
                )
    return (result + result.T) * 0.5


def overlap_coordinates(q, channels, relative_floor=1e-8):
    """B=|Q|^(1/2), W=B^-1; eigenvalue floor applies before square root."""
    q = np.asarray(q, dtype=np.float64)
    invariant = scalar_projection(q, channels)
    if not np.allclose(q, invariant, atol=1e-10, rtol=1e-10):
        raise ValueError("Native augmentation overlap is not rotationally invariant")
    values, vectors = np.linalg.eigh(invariant)
    largest = float(np.max(abs(values)))
    if not np.isfinite(largest) or largest <= 0 or not 0 < relative_floor < 1:
        raise ValueError(
            "Nonzero finite overlap and a positive relative floor required"
        )
    floor = largest * relative_floor
    magnitudes = np.maximum(abs(values), floor)
    decode = (vectors * np.sqrt(magnitudes)[None, :]) @ vectors.T
    encode = (vectors / np.sqrt(magnitudes)[None, :]) @ vectors.T
    return (
        decode,
        encode,
        dict(
            augmentation_eigenvalues=values.tolist(),
            eigenvalue_floor=floor,
            floored_eigenvalues=int(np.sum(abs(values) < floor)),
            scaling_condition_number=float(
                np.sqrt(magnitudes.max() / magnitudes.min())
            ),
        ),
    )


class ScaledPAWHead(nn.Module):
    def __init__(self, features, channels, decode, origin, scale):
        super().__init__()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Positive encoded target scale required")
        self.core = SpeciesPAWHead(features, channels)
        self.register_buffer("decode", torch.as_tensor(decode, dtype=torch.float64))
        self.register_buffer("origin", torch.as_tensor(origin, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))

    def physical(self, encoded):
        matrix = encoded.to(torch.float64) * self.scale + self.origin
        result = self.decode @ matrix @ self.decode.T
        return (result + result.transpose(-1, -2)) * 0.5

    def forward(self, features):
        return self.physical(self.core(features))


class ConditionedPAWModel(nn.Module):
    """Conditioned PAW heads, with frozen-base repair or joint optimization."""

    def __init__(self, base, specifications, train_base=False):
        super().__init__()
        from e3nn import o3

        self.train_base = bool(train_base)
        self.base = base.eval()
        self.base.requires_grad_(self.train_base)
        # The old matrix heads are replaced and never participate in this model.
        for name, parameter in self.base.named_parameters():
            if name.startswith("onsite_"):
                parameter.requires_grad_(False)
        self.config = base.config
        features = sum(
            (p.linear.irreps_out for p in base.local.backbone.products), o3.Irreps()
        )
        self.specifications = specifications
        self.heads = nn.ModuleDict(
            {
                str(z): ScaledPAWHead(
                    features,
                    entry["channels"],
                    entry["decode"],
                    entry["origin"],
                    entry["scale"],
                )
                for z, entry in specifications.items()
            }
        )
        self.matrix_size = max(
            sum(2 * l + 1 for l in e["channels"]) for e in specifications.values()
        )

    @property
    def correction(self):
        return self.base.correction

    @property
    def field_scale(self):
        return self.base.field_scale

    def features(self, graph):
        if self.train_base:
            return self.base.local.backbone(
                graph, training=self.training, compute_force=False
            )["node_feats"]
        self.base.eval()
        with torch.no_grad():
            return self.base.local.backbone(graph, training=False, compute_force=False)[
                "node_feats"
            ]

    def readouts(self, graph):
        features = self.features(graph)
        packed = self.base.local.coefficients(features)
        coeff = torch.cat(
            [
                packed[:, s].reshape(len(features), self.config.radial_count, ir.dim)
                for (_, ir), s in zip(
                    self.base.local.coefficient_irreps,
                    self.base.local.coefficient_irreps.slices(),
                )
            ],
            dim=-1,
        )
        offset = self.base.local.offset(features).mean()
        d = torch.zeros(
            len(features),
            self.matrix_size,
            self.matrix_size,
            dtype=torch.float64,
            device=features.device,
        )
        species = graph["node_attrs"].argmax(-1)
        encoded = {}
        for i, z in enumerate(self.config.atomic_numbers):
            ids = torch.where(species == i)[0]
            head = self.heads[str(z)]
            encoded[str(z)] = head.core(features[ids])
            part = head.physical(encoded[str(z)])
            d[ids, : part.shape[-1], : part.shape[-1]] = part
        return coeff * self.base.field_scale, offset * self.base.field_scale, d, encoded

    def coefficients(self, graph):
        return self.readouts(graph)[:3]

    def forward(self, item, return_encoded=False):
        coeff, offset, d, encoded = self.readouts(item["graph"])
        field, d = PAWModel.fields_from_coefficients(self, item, coeff, offset, d)
        return (field, d, encoded) if return_encoded else (field, d)
