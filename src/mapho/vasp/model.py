"""Joint MACE potential/PAW readouts with native real s/p/d projector conventions.

The model supports nonmagnetic species with complete real s/p/d shells.
It predicts both corrections to a fixed atomic-density VASP baseline. VASP
retains the projectors and overlap. No target-derived field is a model input.
"""

from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .._e3nn import o3

from ..field_model import FieldInputs, FieldModelConfig, LocalMACEField, mace_graph
from ..exchange_ablation import ControlledCorrection
from ..probes import PeriodicProbeGrid


def native_sp_order(channels):
    """Native VASP s; p_y,p_z,p_x versus e3nn s; p_x,p_y,p_z.

    Verified against VASP 6.6.1 SETYLM and the native rotation test. Repeated
    radial channels keep their original order. Unsupported l values fail.
    """
    result = []
    start = 0
    for ell in channels:
        if ell not in (0, 1):
            raise ValueError(
                "This PAW head supports complete s/p projector channels only"
            )
        result.extend([start] if ell == 0 else [start + 1, start + 2, start])
        start += 2 * ell + 1
    return result


def native_projector_transform(channels):
    """Orthogonal e3nn -> VASP real harmonics (SETYLM convention).

    Native d order: xy, yz, (3z²-r²), xz, (x²-y²). Unlike p, d needs
    a linear combination as well as a permutation. Radial order is retained.
    """
    blocks = []
    for ell in channels:
        if ell == 0:
            block = torch.ones(1, 1)
        elif ell == 1:
            block = torch.eye(3)[[1, 2, 0]]
        elif ell == 2:
            block = torch.tensor(
                [
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, -0.5, 0, np.sqrt(3) / 2],
                    [1, 0, 0, 0, 0],
                    [0, 0, -np.sqrt(3) / 2, 0, -0.5],
                ],
                dtype=torch.get_default_dtype(),
            )
        else:
            raise ValueError("PAW readout supports complete s/p/d shells only")
        blocks.append(block)
    if not blocks:
        raise ValueError("Empty projector channels")
    return torch.block_diag(*blocks)


class SpeciesPAWHead(nn.Module):
    def __init__(self, features, channels):
        super().__init__()
        orbitals = o3.Irreps([(1, (ell, (-1) ** ell)) for ell in channels])
        reduced = o3.ReducedTensorProducts("ij=ji", i=orbitals)
        transform = native_projector_transform(channels)
        basis = torch.einsum(
            "ia,kab,jb->kij", transform, reduced.change_of_basis, transform
        )
        basis = (basis + basis.transpose(-1, -2)) * 0.5
        self.register_buffer("matrix_basis", basis)
        self.linear = o3.Linear(features, reduced.irreps_out, biases=True)
        self.product = o3.FullyConnectedTensorProduct(
            features, features, reduced.irreps_out
        )
        for module in (self.linear, self.product):
            for parameter in module.parameters():
                nn.init.zeros_(parameter)

    def forward(self, features):
        coupled = self.linear(features) + self.product(features, features)
        return torch.einsum("ak,kij->aij", coupled, self.matrix_basis)


def gaussian_atoms(numbers, positions, cell, shape, sigma=0.65):
    """Periodic geometric envelope, integral = atom count; no electronic labels."""
    axes = [np.fft.fftfreq(n) * n for n in shape]
    modes = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    q = 2 * np.pi * np.einsum("ab,ijkb->ijka", np.linalg.inv(cell), modes)
    frac = positions @ np.linalg.inv(cell)
    structure = np.zeros(shape, dtype=np.complex128)
    for position in frac:
        structure += np.exp(-2j * np.pi * np.einsum("...a,a->...", modes, position))
    result = np.fft.ifftn(
        structure * np.exp(-0.5 * sigma**2 * np.sum(q * q, axis=-1))
    ).real
    return result * np.prod(shape) / np.linalg.det(cell)


def atomic_inputs(numbers, positions, cell, baseline_potential):
    # FieldInputs is reused as a geometry container. Its historical hxc name
    # here holds the TOTAL atomic-reference potential, not a converged Hxc.
    envelope = gaussian_atoms(numbers, positions, cell, baseline_potential.shape)
    return FieldInputs(numbers, positions, cell, envelope, baseline_potential)


def point_basis(inputs, fractions, config):
    """Small cached decoder basis at selected native-grid points."""
    fractions = np.asarray(fractions)
    if (
        fractions.ndim != 2
        or fractions.shape[1] != 3
        or not np.isfinite(fractions).all()
    ):
        raise ValueError("Finite fractional point coordinates required")
    grid = PeriodicProbeGrid(inputs, (1, 1, 1), config)
    result = torch.zeros(
        len(fractions), len(inputs.numbers), config.radial_count, (config.lmax + 1) ** 2
    )
    for start in range(0, len(fractions), 1024):
        points = (fractions[start : start + 1024] % 1) @ inputs.cell
        neighbors = grid.tree.query_ball_point(
            points, config.probe_cutoff, return_sorted=True
        )
        counts = np.array([len(n) for n in neighbors])
        if not counts.sum():
            continue
        p = np.repeat(np.arange(len(points)), counts)
        images = np.concatenate(neighbors).astype(int)
        vectors = points[p] - grid.locations[images]
        weights = grid.basis(vectors, dtype=torch.float32, device="cpu")
        result.index_put_(
            (torch.as_tensor(p + start), torch.as_tensor(grid.atom_indices[images])),
            weights,
            accumulate=True,
        )
    return result


def periodic_interpolate(field, fractions):
    """Differentiable periodic trilinear interpolation in fractional coordinates."""
    shape = field.new_tensor(field.shape)
    x = (fractions % 1) * shape
    lower = torch.floor(x).long()
    w = x - lower
    result = field.new_zeros(len(fractions))
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                idx = (lower + lower.new_tensor([i, j, k])) % shape.long()
                weight = (
                    (w[:, 0] if i else 1 - w[:, 0])
                    * (w[:, 1] if j else 1 - w[:, 1])
                    * (w[:, 2] if k else 1 - w[:, 2])
                )
                result = result + field[idx[:, 0], idx[:, 1], idx[:, 2]] * weight
    return result


class PAWModel(nn.Module):
    def __init__(
        self,
        config=None,
        mode="full",
        channels=(0, 0, 1, 1),
        avg_neighbors=16.0,
        correction_activation="silu",
        species_channels=None,
    ):
        super().__init__()
        if mode not in ("local", "none", "full"):
            raise ValueError("Modes: local MACE only, none (spectral off), full FNO")
        self.config = config or FieldModelConfig(grid_size=32)
        self.mode = mode
        self.channels = tuple(channels)
        self.avg_neighbors = float(avg_neighbors)
        self.species_channels = (
            None
            if species_channels is None
            else {int(z): tuple(ls) for z, ls in species_channels.items()}
        )
        if self.species_channels is not None and set(self.species_channels) != set(
            self.config.atomic_numbers
        ):
            raise ValueError("PAW projector species must match MACE elements")
        if correction_activation not in ("silu", "gelu"):
            raise ValueError("Correction activation must be silu or gelu")
        self.correction_activation = correction_activation
        self.local = LocalMACEField(self.config)
        for interaction in self.local.backbone.interactions:
            interaction.avg_num_neighbors = self.avg_neighbors
        features = sum(
            (p.linear.irreps_out for p in self.local.backbone.products), o3.Irreps()
        )
        if self.species_channels is None:
            orbitals = o3.Irreps([(1, (ell, (-1) ** ell)) for ell in channels])
            reduced = o3.ReducedTensorProducts("ij=ji", i=orbitals)
            order = native_sp_order(channels)
            basis = reduced.change_of_basis[:, order][:, :, order]
            self.register_buffer("matrix_basis", basis)
            self.onsite_linear = o3.Linear(features, reduced.irreps_out, biases=True)
            # Products supply, in particular, axial 1e components allowed between
            # different radial p channels. No symmetric matrix component is omitted.
            self.onsite_product = o3.FullyConnectedTensorProduct(
                features, features, reduced.irreps_out
            )
        else:
            self.onsite_heads = nn.ModuleDict(
                {
                    str(z): SpeciesPAWHead(features, ls)
                    for z, ls in self.species_channels.items()
                }
            )
            self.matrix_size = max(
                sum(2 * l + 1 for l in ls) for ls in self.species_channels.values()
            )
        self.correction = (
            None
            if mode == "local"
            else ControlledCorrection(
                self.config,
                "none" if mode == "none" else "full",
                activation=correction_activation,
            )
        )
        self.register_buffer("field_scale", torch.ones(()))
        self.register_buffer("onsite_scale", torch.ones(()))
        heads = [self.local.coefficients, self.local.offset]
        if self.species_channels is None:
            heads.extend([self.onsite_linear, self.onsite_product])
        for head in heads:
            for p in head.parameters():
                nn.init.zeros_(p)

    def coefficients(self, graph):
        features = self.local.backbone(
            graph, training=self.training, compute_force=False
        )["node_feats"]
        packed = self.local.coefficients(features)
        coeff = torch.cat(
            [
                packed[:, s].reshape(len(features), self.config.radial_count, ir.dim)
                for (_, ir), s in zip(
                    self.local.coefficient_irreps,
                    self.local.coefficient_irreps.slices(),
                )
            ],
            dim=-1,
        )
        offset = self.local.offset(features).mean()
        if self.species_channels is None:
            coupled = self.onsite_linear(features) + self.onsite_product(
                features, features
            )
            d = torch.einsum("ak,kij->aij", coupled, self.matrix_basis)
        else:
            d = features.new_zeros(len(features), self.matrix_size, self.matrix_size)
            species = graph["node_attrs"].argmax(dim=-1)
            for i, z in enumerate(self.config.atomic_numbers):
                ids = torch.where(species == i)[0]
                part = self.onsite_heads[str(z)](features[ids])
                d[ids, : part.shape[-1], : part.shape[-1]] = part
        d = d * self.onsite_scale
        return coeff * self.field_scale, offset * self.field_scale, d

    def forward(self, item):
        return self.fields_from_coefficients(item, *self.coefficients(item["graph"]))

    def fields_from_coefficients(self, item, coeff, offset, d):
        probes = item["probes"]
        local = (
            probes.contract(coeff)
            if isinstance(probes, PeriodicProbeGrid)
            else torch.einsum("parl,arl->p", probes, coeff)
        )
        field = (local + offset).reshape(item["shape"])
        correction = torch.zeros_like(field)
        if self.correction is not None:
            channels = torch.stack([field, item["baseline"], item["envelope"]])
            correction = self.correction(channels, item["cell"]) * self.field_scale
        if "point_probes" in item:
            native = torch.einsum("parl,arl->p", item["point_probes"], coeff) + offset
            return native + periodic_interpolate(correction, item["fractions"]), d
        return field + correction, d

    def description(self):
        return dict(
            config=asdict(self.config),
            mode=self.mode,
            channels=list(self.channels),
            avg_neighbors=self.avg_neighbors,
            species_channels=self.species_channels,
            correction_activation=self.correction_activation,
            inputs="geometry, total fixed-atomic-density VASP potential, geometric Gaussian atom envelope",
            outputs="delta total smooth potential and delta full symmetric native PAW D",
            symmetry="O(3) equivariant on-site matrices; scalar periodic potential",
            gauge="Unshifted native VASP gauge for baseline and target; no label mean supplied at inference",
        )
