"""MACE atom-to-probe scalar potentials and a periodic metric-aware FNO residual.

All meshes use fractional lattice axes (i,j,k), with row-vector cells in A.
The input object deliberately excludes converged electronic quantities.
"""

from .schema import canonical_schema

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import torch
from torch import nn
from e3nn import o3
from ase import Atoms
from scipy.signal import resample
from torch.utils.checkpoint import checkpoint

from .probes import PeriodicProbeGrid


def resize_field(field, shape):
    """Periodic Fourier resampling, including correct even-grid Nyquist handling."""
    result = np.asarray(field)
    if result.ndim != 3 or len(shape) != 3 or min(shape) < 2:
        raise ValueError("Expected a 3D periodic grid and three dimensions >= 2")
    for axis, size in enumerate(shape):
        if result.shape[axis] != size:
            result = resample(result, size, axis=axis)
    return result.real.copy()


@dataclass
class FieldInputs:
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    reference_density: np.ndarray
    reference_hxc: np.ndarray

    @classmethod
    def from_sample(cls, sample):
        return cls(
            sample.numbers.copy(),
            sample.positions.copy(),
            sample.cell.copy(),
            sample.fields["reference_density"].copy(),
            sample.fields["reference_hxc"].copy(),
        )

    def validate(self):
        if (
            self.positions.shape != (len(self.numbers), 3)
            or self.cell.shape != (3, 3)
            or np.linalg.det(self.cell) <= 1e-10
            or self.reference_density.ndim != 3
            or min(self.reference_density.shape) < 2
            or self.reference_hxc.shape != self.reference_density.shape
        ):
            raise ValueError("Invalid periodic field input dimensions")
        if not all(
            np.isfinite(a).all()
            for a in (
                self.positions,
                self.cell,
                self.reference_density,
                self.reference_hxc,
            )
        ):
            raise ValueError("Non-finite field input")


@dataclass
class FieldModelConfig:
    atomic_numbers: tuple = (14,)
    atom_cutoff: float = 4.5
    probe_cutoff: float = 3.0
    channels: int = 8
    radial_count: int = 12
    lmax: int = 2
    correlation: int = 3
    grid_size: int = 16
    grid_shape: tuple | None = None
    fno_width: int = 8
    fno_layers: int = 2
    q_cutoff: float = 6.0  # inverse angstrom; spherical physical frequency support
    probe_chunk_size: int = 512
    probe_edge_chunk_size: int = 2048
    spectral_chunk_size: int = 256

    def validate(self):
        if self.grid_shape is not None and (
            len(self.grid_shape) != 3
            or any(int(n) != n or n < 4 for n in self.grid_shape)
        ):
            raise ValueError("Grid shape must contain three integers >= 4")
        if any(
            int(n) != n or n < 1
            for n in (
                self.probe_chunk_size,
                self.probe_edge_chunk_size,
                self.spectral_chunk_size,
            )
        ):
            raise ValueError("Memory chunk sizes must be positive integers")
        if (
            not self.atomic_numbers
            or len(set(self.atomic_numbers)) != len(self.atomic_numbers)
            or any(int(z) != z or z < 1 or z > 118 for z in self.atomic_numbers)
            or min(self.atom_cutoff, self.probe_cutoff, self.q_cutoff) <= 0
            or self.lmax not in (0, 1, 2)
            or self.channels < 1
            or self.radial_count < 2
            or self.correlation < 1
            or self.grid_size < 4
            or self.fno_width < 1
            or self.fno_layers < 1
        ):
            raise ValueError("Invalid field model configuration")


def mace_graph(inputs, config):
    from mace.data import AtomicData, config_from_atoms
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    inputs.validate()
    if not set(inputs.numbers).issubset(config.atomic_numbers):
        raise ValueError("Input contains an element absent from the model")
    atoms = Atoms(
        inputs.numbers, positions=inputs.positions, cell=inputs.cell, pbc=True
    )
    data = AtomicData.from_config(
        config_from_atoms(atoms),
        z_table=AtomicNumberTable(config.atomic_numbers),
        cutoff=config.atom_cutoff,
    )
    return Batch.from_data_list([data]).to_dict()


def probe_basis(inputs, shape, config, dtype=torch.float32):
    """Sum ALL periodic images, not just the nearest image, into radial harmonics.

    Returns (grid points, atoms, radial channels, angular components).
    Diagnostic dense reference only, limited to 256 MiB. Production uses
    PeriodicProbeGrid. Geometry gradients are not supported.
    """
    inputs.validate()
    size_bytes = (
        int(np.prod(shape))
        * len(inputs.numbers)
        * config.radial_count
        * (config.lmax + 1) ** 2
        * torch.empty((), dtype=dtype).element_size()
    )
    if size_bytes > 256 * 1024**2:
        raise ValueError("Dense probe reference exceeds 256 MiB; use PeriodicProbeGrid")
    frac = np.stack(np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), -1)
    grid = frac.reshape(-1, 3) @ inputs.cell
    inverse = np.linalg.inv(inputs.cell)
    positions = (inputs.positions @ inverse % 1) @ inputs.cell
    bounds = (
        np.ceil(config.probe_cutoff * np.linalg.norm(inverse, axis=0)).astype(int) + 1
    )
    shifts = np.array(list(product(*(range(-b, b + 1) for b in bounds)))) @ inputs.cell
    nangular = (config.lmax + 1) ** 2
    basis = torch.zeros(
        len(grid), len(positions), config.radial_count, nangular, dtype=dtype
    )
    centers = torch.linspace(0, config.probe_cutoff, config.radial_count, dtype=dtype)
    width = config.probe_cutoff / (config.radial_count - 1)
    for atom, pos in enumerate(positions):
        # Bound memory independently of the number of periodic images.
        for start in range(0, len(shifts), 16):
            vectors = grid[None] - pos - shifts[start : start + 16, None]
            distance = np.linalg.norm(vectors, axis=-1)
            image_id, point_id = np.where(distance < config.probe_cutoff)
            if len(point_id) == 0:
                continue
            v = torch.as_tensor(vectors[image_id, point_id], dtype=dtype)
            r = torch.as_tensor(distance[image_id, point_id], dtype=dtype)
            sh = o3.spherical_harmonics(
                list(range(config.lmax + 1)),
                v,
                normalize=True,
                normalization="component",
            )
            # r**l removes the angular singularity at a probe on a nucleus.
            for ell in range(1, config.lmax + 1):
                sh[:, ell**2 : (ell + 1) ** 2] *= (r / config.probe_cutoff)[
                    :, None
                ] ** ell
            radial = torch.exp(-0.5 * ((r[:, None] - centers) / width) ** 2)
            radial *= (1 - (r / config.probe_cutoff) ** 2)[:, None] ** 2
            basis[:, atom].index_add_(
                0, torch.as_tensor(point_id), radial[:, :, None] * sh[:, None]
            )
    return basis


class LocalMACEField(nn.Module):
    """Trainable MACE many-body features contracted with radial probe harmonics."""

    def __init__(self, config=None):
        super().__init__()
        from mace.modules.models import MACE
        from mace.modules.blocks import (
            RealAgnosticInteractionBlock,
            RealAgnosticResidualInteractionBlock,
        )

        self.config = config or FieldModelConfig()
        c = self.config
        c.validate()
        hidden = o3.Irreps([(c.channels, (l, (-1) ** l)) for l in range(c.lmax + 1)])
        self.backbone = MACE(
            r_max=c.atom_cutoff,
            num_bessel=6,
            num_polynomial_cutoff=5,
            max_ell=c.lmax,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=len(c.atomic_numbers),
            hidden_irreps=hidden,
            MLP_irreps=o3.Irreps(f"{c.channels}x0e"),
            atomic_energies=np.zeros((1, len(c.atomic_numbers))),
            avg_num_neighbors=16,
            atomic_numbers=list(c.atomic_numbers),
            correlation=c.correlation,
            gate=torch.nn.functional.silu,
            radial_MLP=[32, 32],
        )
        # Energy readouts are not part of this potential-field objective.
        self.backbone.readouts.requires_grad_(False)
        feature_irreps = sum(
            (p.linear.irreps_out for p in self.backbone.products), o3.Irreps()
        )
        self.coefficient_irreps = o3.Irreps(
            [(c.radial_count, (l, (-1) ** l)) for l in range(c.lmax + 1)]
        )
        self.coefficients = o3.Linear(
            feature_irreps, self.coefficient_irreps, biases=True
        )
        self.offset = o3.Linear(feature_irreps, "1x0e", biases=True)

    def atomic_coefficients(self, graph):
        features = self.backbone(graph, training=self.training, compute_force=False)[
            "node_feats"
        ]
        packed = self.coefficients(features)
        # e3nn stores multiplicities within each angular momentum block.
        coeff = torch.cat(
            [
                packed[:, s].reshape(len(features), self.config.radial_count, ir.dim)
                for (_, ir), s in zip(
                    self.coefficient_irreps, self.coefficient_irreps.slices()
                )
            ],
            dim=-1,
        )
        return coeff, self.offset(features).mean()

    def forward(self, graph, probes):
        coeff, offset = self.atomic_coefficients(graph)
        if isinstance(probes, PeriodicProbeGrid):
            return probes.contract(coeff) + offset
        # Small dense inputs remain supported for diagnostic comparisons.
        return torch.einsum("parl,arl->p", probes, coeff) + offset


class MetricSpectralLayer(nn.Module):
    """Real radial channel mixing in physical reciprocal space; conjugacy exact."""

    def __init__(self, width, q_cutoff, chunk_size=256):
        super().__init__()
        self.width, self.q_cutoff = width, q_cutoff
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError("Spectral chunk size must be a positive integer")
        self.chunk_size = chunk_size
        self.radial = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, width * width)
        )
        self.pointwise = nn.Conv3d(width, width, 1)

    def _mix(self, spectrum, q2):
        weights = self.radial(torch.log1p(q2)[:, None]).reshape(
            -1, self.width, self.width
        )
        weights = (
            weights
            * (1 - q2 / self.q_cutoff**2).square()[:, None, None]
            / self.width**0.5
        )
        # Keep filters real; never materialize a complex channel-mixing tensor.
        return torch.complex(
            torch.einsum("bcm,mcd->bdm", spectrum.real, weights),
            torch.einsum("bcm,mcd->bdm", spectrum.imag, weights),
        )

    def activate(self, x):
        return torch.nn.functional.silu(x)

    def forward(self, x, cell):
        shape = x.shape[-3:]
        axes = [torch.fft.fftfreq(n, device=x.device, dtype=x.dtype) * n for n in shape]
        modes = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
        # q = 2pi A^-1 n (column convention), since A has direct vectors as rows.
        q = 2 * torch.pi * torch.einsum("ab,ijkb->ijka", torch.linalg.inv(cell), modes)
        q2 = q.square().sum(-1)
        # Exclude Nyquist planes: their negative-frequency partners are ambiguous
        # in a skew cell. This also guarantees real output before .real is taken.
        support = q2 < self.q_cutoff**2
        for axis, n in enumerate(shape):
            if n % 2 == 0:
                support &= modes[..., axis] != -n // 2
        spectrum = torch.fft.fftn(x, dim=(-3, -2, -1))
        indices = support.flatten().nonzero().flatten()
        spectrum_flat = spectrum.flatten(2)
        q2_flat = q2.flatten()
        pieces = []
        for ids in indices.split(self.chunk_size):
            part, frequencies = spectrum_flat[:, :, ids], q2_flat[ids]
            if torch.is_grad_enabled():
                # Recompute mixing matrices in backward, retaining only O(C) data
                # per mode instead of O(C^2) filters across every mode and layer.
                mixed = checkpoint(
                    self._mix,
                    part,
                    frequencies,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                mixed = self._mix(part, frequencies)
            pieces.append(mixed)
        filtered = torch.zeros_like(spectrum_flat)
        filtered.index_copy_(2, indices, torch.cat(pieces, dim=-1))
        filtered = filtered.reshape_as(spectrum)
        return self.activate(
            torch.fft.ifftn(filtered, dim=(-3, -2, -1)).real + self.pointwise(x)
        )


class ResidualFNO(nn.Module):
    """Potential correction from the local prediction and atomic reference fields."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or FieldModelConfig()
        c = self.config
        c.validate()
        self.register_buffer("input_scale", torch.ones(3))
        self.register_buffer("output_scale", torch.ones(()))
        self.lift = nn.Conv3d(3, c.fno_width, 1)
        self.layers = nn.ModuleList(
            [
                MetricSpectralLayer(c.fno_width, c.q_cutoff, c.spectral_chunk_size)
                for _ in range(c.fno_layers)
            ]
        )
        self.readout = nn.Conv3d(c.fno_width, 1, 1)

    def forward(self, channels, cell):
        x = self.lift(channels[None] / self.input_scale[None, :, None, None, None])
        for layer in self.layers:
            x = layer(x, cell)
        return self.readout(x)[0, 0] * self.output_scale


def predict_fields(local, fno, inputs):
    """Full-resolution local decoder plus Fourier-interpolated global correction.

    Never accepts converged densities, potentials, eigenvalues, or target means.
    """
    c = local.config
    dtype = next(local.parameters()).dtype
    graph = mace_graph(inputs, c)
    shape = inputs.reference_hxc.shape
    local.eval()
    if fno is not None:
        fno.eval()
    with torch.no_grad():
        probes = PeriodicProbeGrid(inputs, shape, c)
        coefficients, offset = local.atomic_coefficients(graph)
        delta = (probes.contract(coefficients) + offset).reshape(shape).numpy()
        if fno is None:
            return inputs.reference_hxc + delta
        coarse = (c.grid_size,) * 3
        coarse_delta = (
            (probes.with_shape(coarse).contract(coefficients) + offset)
            .reshape(coarse)
            .numpy()
        )
        channels = torch.as_tensor(
            np.stack(
                [
                    coarse_delta,
                    resize_field(inputs.reference_density, coarse),
                    resize_field(inputs.reference_hxc, coarse),
                ]
            ),
            dtype=dtype,
        )
        correction = fno(channels, torch.as_tensor(inputs.cell, dtype=dtype)).numpy()
    return inputs.reference_hxc + delta + resize_field(correction, shape)


def save_field_models(path, local, fno, metadata):
    torch.save(
        {
            "schema": "mapho.field_model.v1",
            "config": asdict(local.config),
            "local": local.state_dict(),
            "fno": fno.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_field_models(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if canonical_schema(checkpoint["schema"]) != "mapho.field_model.v1":
        raise ValueError("Unsupported field model checkpoint")
    config = FieldModelConfig(**checkpoint["config"])
    local, fno = LocalMACEField(config), ResidualFNO(config)
    local.load_state_dict(checkpoint["local"])
    fno.load_state_dict(checkpoint["fno"])
    return local, fno, checkpoint["metadata"]
