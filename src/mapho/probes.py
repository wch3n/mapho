"""Periodic probe evaluation with bounded neighbor batches and recomputed gradients."""

import copy
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
import torch
from e3nn import o3
from torch.autograd.function import once_differentiable


def _shape(shape):
    if len(shape) != 3 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("Expected three positive integer probe dimensions")
    return tuple(map(int, shape))


class PeriodicProbeGrid:
    """Fixed geometry, a periodic image tree, and a lazily evaluated fractional mesh.

    Only images within cutoff of the cell's fractional bounding box are indexed.
    The reciprocal plane heights give conservative bounds even for skew cells.
    No electronic fields or point-by-atom arrays are retained.
    """

    def __init__(self, inputs, shape, config):
        inputs.validate()
        config.validate()
        self.shape = _shape(shape)
        self.cell = inputs.cell.copy()
        self.natoms = len(inputs.numbers)
        self.cutoff = config.probe_cutoff
        self.radial_count = config.radial_count
        self.lmax = config.lmax
        self.probe_chunk_size = config.probe_chunk_size
        self.edge_chunk_size = config.probe_edge_chunk_size
        inverse = np.linalg.inv(self.cell)
        fractional = inputs.positions @ inverse % 1
        margin = self.cutoff * np.linalg.norm(inverse, axis=0)
        bounds = np.ceil(margin).astype(int) + 1
        locations, atoms = [], []
        for shift in product(*(range(-b, b + 1) for b in bounds)):
            translated = fractional + shift
            keep = np.all((translated >= -margin) & (translated <= 1 + margin), axis=1)
            if np.any(keep):
                locations.append(translated[keep] @ self.cell)
                atoms.append(np.flatnonzero(keep))
        self.locations = np.concatenate(locations)
        self.atom_indices = np.concatenate(atoms)
        self.tree = cKDTree(self.locations)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def with_shape(self, shape):
        result = copy.copy(self)
        result.shape = _shape(shape)
        return result

    def edges(self):
        """Yield global probe indices, atom indices and vectors in bounded batches."""
        for start in range(0, self.size, self.probe_chunk_size):
            flat = np.arange(start, min(start + self.probe_chunk_size, self.size))
            fractional = (
                np.stack(np.unravel_index(flat, self.shape), axis=-1) / self.shape
            )
            points = fractional @ self.cell
            neighbors = self.tree.query_ball_point(
                points, self.cutoff, return_sorted=True
            )
            counts = np.fromiter((len(n) for n in neighbors), dtype=np.int64)
            if not counts.sum():
                continue
            probes = np.repeat(np.arange(len(points)), counts)
            images = np.concatenate(neighbors).astype(np.int64, copy=False)
            for edge_start in range(0, len(images), self.edge_chunk_size):
                p = probes[edge_start : edge_start + self.edge_chunk_size]
                a = images[edge_start : edge_start + self.edge_chunk_size]
                vectors = points[p] - self.locations[a]
                keep = np.linalg.norm(vectors, axis=-1) < self.cutoff
                if np.any(keep):
                    yield p[keep] + start, self.atom_indices[a[keep]], vectors[keep]

    def basis(self, vectors, *, dtype, device):
        v = torch.as_tensor(vectors, dtype=dtype, device=device)
        # Match the original decoder's double-precision geometry calculation.
        r = torch.as_tensor(
            np.linalg.norm(vectors, axis=-1), dtype=dtype, device=device
        )
        sh = o3.spherical_harmonics(
            list(range(self.lmax + 1)), v, normalize=True, normalization="component"
        )
        for ell in range(1, self.lmax + 1):
            sh[:, ell**2 : (ell + 1) ** 2] *= (r / self.cutoff)[:, None] ** ell
        centers = torch.linspace(
            0, self.cutoff, self.radial_count, dtype=dtype, device=device
        )
        width = self.cutoff / (self.radial_count - 1)
        radial = torch.exp(-0.5 * ((r[:, None] - centers) / width) ** 2)
        radial *= (1 - (r / self.cutoff) ** 2)[:, None] ** 2
        return radial[:, :, None] * sh[:, None]

    def contract(self, coefficients):
        expected = (self.natoms, self.radial_count, (self.lmax + 1) ** 2)
        if tuple(coefficients.shape) != expected:
            raise ValueError(f"Expected probe coefficients with shape {expected}")
        return _ProbeContract.apply(coefficients, self)


class _ProbeContract(torch.autograd.Function):
    """Linear contraction; backward rebuilds local bases instead of saving them.

    Supports first derivatives of model parameters. Probe geometry is fixed;
    geometry derivatives and higher-order autograd are deliberately unsupported.
    """

    @staticmethod
    def forward(ctx, coefficients, probes):
        ctx.probes = probes
        ctx.coefficient_shape = coefficients.shape
        result = coefficients.new_zeros(probes.size)
        for points, atoms, vectors in probes.edges():
            weights = probes.basis(
                vectors, dtype=coefficients.dtype, device=coefficients.device
            )
            ids = torch.as_tensor(atoms, device=coefficients.device)
            values = (coefficients[ids] * weights).sum((1, 2))
            result.index_add_(
                0, torch.as_tensor(points, device=coefficients.device), values
            )
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        probes = ctx.probes
        gradient = grad_output.new_zeros(ctx.coefficient_shape)
        for points, atoms, vectors in probes.edges():
            weights = probes.basis(
                vectors, dtype=grad_output.dtype, device=grad_output.device
            )
            ids = torch.as_tensor(points, device=grad_output.device)
            values = grad_output[ids, None, None] * weights
            gradient.index_add_(
                0, torch.as_tensor(atoms, device=grad_output.device), values
            )
        return gradient, None
