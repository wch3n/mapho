"""Fock exchange with a structure-conditioned, linear Hermitian correction.

This pilot uses converged PBE occupied orbitals as a reference, not only atomic
positions. The restricted model is M_g K[PBE] M_g, with a positive real field g.
It is invariant to occupied-orbital unitary rotations at each q and is not a
general parameterization of the hybrid density matrix.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .field_model import LocalMACEField, FieldModelConfig, ResidualFNO


def pw_to_grid(coefficients, mill, shape):
    """PW rows -> orthonormal discrete real-space rows; no physical-volume factor."""
    shape = tuple(map(int, shape))
    if len(shape) != 3 or coefficients.shape[-1] != len(mill):
        raise ValueError("Inconsistent PW/grid shapes")
    index = tuple(np.asarray(mill, int).T % np.asarray(shape)[:, None])
    if len(set(zip(*index))) != len(mill):
        raise ValueError("FFT grid aliases distinct plane waves")
    grid = np.zeros((*coefficients.shape[:-1], *shape), np.complex128)
    grid[(..., *index)] = coefficients
    return np.fft.ifftn(grid, axes=(-3, -2, -1), norm="ortho")


def pw_mask(mill, shape):
    mask = np.zeros(tuple(shape), bool)
    mask[tuple(np.asarray(mill, int).T % np.asarray(shape)[:, None])] = True
    return mask


def screened_kernel(shape, cell_bohr, shift, mu=0.106):
    """Fourier transform of erfc(mu*r)/r, including its finite zero-mode limit."""
    cell = np.asarray(cell_bohr, float)
    if mu <= 0 or not np.isfinite(mu) or np.linalg.det(cell) <= 0:
        raise ValueError("Expected positive screening and a right-handed cell")
    modes = np.stack(
        np.meshgrid(*(np.fft.fftfreq(n) * n for n in shape), indexing="ij"), -1
    )
    g = (
        2
        * np.pi
        * np.einsum("ab,...b->...a", np.linalg.inv(cell), modes + np.asarray(shift))
    )
    g2 = np.square(g).sum(-1)
    result = np.full(tuple(shape), np.pi / mu**2)
    nonzero = g2 > 1e-14
    result[nonzero] = 4 * np.pi * (-np.expm1(-g2[nonzero] / (4 * mu**2))) / g2[nonzero]
    return result


def gygi_baldereschi_zero(cell_bohr, ecutwfc_ry, ecutrho_ry, kmesh):
    """QE 7.2 unscreened zero mode, without gamma extrapolation (bohr^2).

    Geometry/cutoff/mesh only; no hybrid orbitals or target labels enter.
    This ports exx_base.f90's Gaussian auxiliary-function sum, with e2=2
    factored out. It regularizes the reciprocal-space singularity without
    truncating the real-space Coulomb interaction.
    """
    cell = np.asarray(cell_bohr, float)
    mesh = np.asarray(kmesh, int)
    if (
        cell.shape != (3, 3)
        or not np.isfinite(cell).all()
        or np.linalg.det(cell) <= 0
        or mesh.shape != (3,)
        or np.any(mesh < 1)
        or not np.array_equal(mesh, kmesh)
        or not np.isfinite([ecutwfc_ry, ecutrho_ry]).all()
        or ecutwfc_ry <= 0
        or ecutrho_ry < 4 * ecutwfc_ry
    ):
        raise ValueError("Invalid Coulomb auxiliary-function parameters")
    alpha = 10.0 / ecutwfc_ry
    bounds = np.ceil(
        np.sqrt(ecutrho_ry) * np.linalg.norm(cell, axis=1) / (2 * np.pi)
    ).astype(int)
    mill = np.stack(
        np.meshgrid(*(np.arange(-n, n + 1) for n in bounds), indexing="ij"), -1
    ).reshape(-1, 3)
    reciprocal = 2 * np.pi * np.linalg.inv(cell)
    gvectors = mill @ reciprocal.T
    gvectors = gvectors[np.square(gvectors).sum(-1) <= ecutrho_ry * (1 + 1e-12)]
    qpoints = np.stack(
        np.meshgrid(*(np.arange(n) / n for n in mesh), indexing="ij"), -1
    ).reshape(-1, 3)
    total = 0.0
    for q in qpoints @ reciprocal.T:
        q2 = np.square(gvectors + q).sum(-1)
        q2 = q2[q2 > 1e-12]
        total += np.sum(np.exp(-alpha * q2) / q2)
    zero = np.prod(mesh) * np.linalg.det(cell) / np.sqrt(np.pi * alpha) - 4 * np.pi * (
        total - alpha
    )
    if not np.isfinite(zero) or zero <= 0:
        raise ValueError("Nonpositive Coulomb zero mode; unsupported geometry/mesh")
    return float(zero)


def coulomb_kernel(shape, cell_bohr, shift, zero_mode):
    """Full 4*pi/|G+k-q|^2 with an explicit finite-mesh singular correction."""
    cell = np.asarray(cell_bohr, float)
    if not np.isfinite(zero_mode) or zero_mode <= 0 or np.linalg.det(cell) <= 0:
        raise ValueError("Unscreened exchange requires a positive explicit zero mode")
    modes = np.stack(
        np.meshgrid(*(np.fft.fftfreq(n) * n for n in shape), indexing="ij"), -1
    )
    g = (
        2
        * np.pi
        * np.einsum("ab,...b->...a", np.linalg.inv(cell), modes + np.asarray(shift))
    )
    g2 = np.square(g).sum(-1)
    result = np.full(tuple(shape), float(zero_mode))
    nonzero = g2 > 1e-14
    result[nonzero] = 4 * np.pi / g2[nonzero]
    return result


class FockExchange:
    """Apply 0.25 Fock exchange in eV with full q sum and native PW projection.

    Orbitals have shape (nq,nocc,nx,ny,nz) and sum |u|^2 = 1. Weights sum
    to one; occupied orbitals are one spin channel (no spin factor of two).
    """

    def __init__(
        self,
        orbitals,
        kpoints,
        weights,
        cell_bohr,
        masks,
        *,
        alpha=0.25,
        mu=0.106,
        zero_mode=None,
        rydberg_eV=13.605693122994,
        recompute=True,
    ):
        if orbitals.ndim != 5 or not orbitals.is_complex():
            raise ValueError("Expected complex (q,occupied,x,y,z) orbitals")
        if alpha < 0 or not np.isclose(np.sum(weights), 1):
            raise ValueError("Invalid exchange fraction or q weights")
        if len(kpoints) != len(orbitals) or len(weights) != len(orbitals):
            raise ValueError("q-point dimensions do not agree")
        if not np.isfinite(mu) or mu < 0 or (mu == 0 and zero_mode is None):
            raise ValueError("Unscreened exchange requires an explicit zero mode")
        if mu > 0 and zero_mode is not None:
            raise ValueError("A screened kernel supplies its own zero mode")
        self.orbitals, self.recompute = orbitals, recompute
        self.shape, self.kpoints = tuple(orbitals.shape[-3:]), np.asarray(kpoints)
        dtype, device = orbitals.real.dtype, orbitals.device
        self.masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
        self.kernels = []
        prefactor = (
            2 * alpha * rydberg_eV * np.prod(self.shape) / np.linalg.det(cell_bohr)
        )
        for k in self.kpoints:
            kernel = (
                (lambda shift: screened_kernel(self.shape, cell_bohr, shift, mu))
                if mu > 0
                else (
                    lambda shift: coulomb_kernel(
                        self.shape, cell_bohr, shift, zero_mode
                    )
                )
            )
            values = np.stack(
                [
                    prefactor * weight * kernel(k - q)
                    for q, weight in zip(self.kpoints, weights)
                ]
            )
            self.kernels.append(torch.as_tensor(values, dtype=dtype, device=device))

    def project(self, psi, k):
        return torch.fft.ifftn(
            torch.fft.fftn(psi, dim=(-3, -2, -1)) * self.masks[k], dim=(-3, -2, -1)
        )

    @staticmethod
    def _q_action(psi, phi, kernel):
        pairs = phi.conj()[:, None] * psi[None]
        convolutions = torch.fft.ifftn(
            kernel * torch.fft.fftn(pairs, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        )
        return -(phi[:, None] * convolutions).sum(0)

    def apply(self, psi, k, gate=None):
        if psi.shape[-3:] != self.shape or not psi.is_complex():
            raise ValueError("Invalid wavefunction grid")
        if gate is not None and (gate.shape != self.shape or gate.is_complex()):
            raise ValueError("Gate must be a real field on the wavefunction grid")
        right = self.project(psi, k)
        if gate is not None:
            right = right * gate
        result = torch.zeros_like(right)
        for phi, kernel in zip(self.orbitals, self.kernels[k]):
            if self.recompute and torch.is_grad_enabled() and right.requires_grad:
                value = checkpoint(
                    self._q_action,
                    right,
                    phi,
                    kernel,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = self._q_action(right, phi, kernel)
            result = result + value
        if gate is not None:
            result = result * gate
        return self.project(result, k)


# Retain the original public name for existing HSE scripts and checkpoints.
ScreenedExchange = FockExchange


class ExchangeGate(nn.Module):
    """MACE alone, MACE plus pointwise control, or MACE plus metric FNO.

    Network nonlinearity only predicts g from structure/reference fields. No
    wavefunction being acted upon is an input to this coefficient predictor.
    """

    def __init__(self, variant="fno", config=None):
        super().__init__()
        if variant not in ("mace", "pointwise", "fno"):
            raise ValueError("Unknown exchange model")
        self.variant = variant
        self.config = config or FieldModelConfig(fno_width=4, fno_layers=2)
        self.local = LocalMACEField(self.config)
        with torch.no_grad():
            for module in (self.local.coefficients, self.local.offset):
                for parameter in module.parameters():
                    parameter.zero_()
        if variant == "fno":
            self.correction = ResidualFNO(self.config)
            with torch.no_grad():
                self.correction.input_scale.copy_(torch.tensor([1.0, 0.2, 20.0]))
                self.correction.readout.weight.zero_()
                self.correction.readout.bias.zero_()
        elif variant == "pointwise":
            self.correction = nn.Sequential(
                nn.Conv3d(3, 16, 1),
                nn.SiLU(),
                nn.Conv3d(16, 16, 1),
                nn.SiLU(),
                nn.Conv3d(16, 16, 1),
                nn.SiLU(),
                nn.Conv3d(16, 1, 1),
            )
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def forward(self, graph, probes, cell, density, hxc):
        local = self.local(graph, probes).reshape(density.shape)
        if self.variant == "mace":
            field = local
        else:
            channels = torch.stack([local, density, hxc])
            if self.variant == "fno":
                correction = self.correction(channels, cell)
            else:
                scale = channels.new_tensor([1.0, 0.2, 20.0])[:, None, None, None]
                correction = self.correction((channels / scale)[None])[0, 0]
            field = local + correction
        return torch.exp(0.1 * torch.tanh(field))
