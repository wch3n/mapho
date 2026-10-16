"""Controlled correction architectures sharing the same local FNO path.

Legacy field/exchange models and their checkpoints are unchanged. These models
separate spectral filtering, local capacity, and whole-cell average information.
"""

import numpy as np
import torch
from torch import nn

from .field_model import FieldInputs, LocalMACEField, MetricSpectralLayer, probe_basis

MODES = ("none", "full", "local", "mean")


class ControlledLayer(MetricSpectralLayer):
    def configure(self, mode, activation="silu"):
        if mode not in MODES:
            raise ValueError("Unknown mixing mode")
        if activation not in ("silu", "gelu"):
            raise ValueError("Correction activation must be silu or gelu")
        self.mode = mode
        self.activation_name = activation
        if mode in ("none", "local"):
            self.radial = None
        if mode == "local":
            # Match the radial network budget with a genuinely pointwise MLP.
            radial_parameters = 32 + 17 * self.width**2
            hidden = max(
                1, round((radial_parameters - self.width) / (2 * self.width + 1))
            )
            self.extra = nn.Sequential(
                nn.Conv3d(self.width, hidden, 1),
                nn.SiLU(),
                nn.Conv3d(hidden, self.width, 1),
            )

    def activate(self, x):
        if self.activation_name == "gelu":
            return torch.nn.functional.gelu(x, approximate="none")
        return super().activate(x)

    def forward(self, x, cell):
        if self.mode == "full":
            return super().forward(x, cell)
        y = self.pointwise(x)
        if self.mode == "mean":
            # Exactly the q=0 part of the original FFT layer, including scaling.
            w = self.radial(x.new_zeros(1, 1)).reshape(self.width, self.width)
            avg = x.mean(dim=(-3, -2, -1))
            y = y + (avg @ w / self.width**0.5)[:, :, None, None, None]
        elif self.mode == "local":
            y = y + self.extra(x) / self.width**0.5
        return self.activate(y)


class ControlledCorrection(nn.Module):
    def __init__(self, config, mode, mean=None, scale=None, *, activation="silu"):
        super().__init__()
        self.config, self.mode = config, mode
        self.register_buffer(
            "input_mean", torch.zeros(3) if mean is None else mean.clone()
        )
        self.register_buffer(
            "input_scale", torch.ones(3) if scale is None else scale.clone()
        )
        # Construct ALL common parameters before installing variant-specific ones.
        # A repeated RNG seed therefore gives identical local paths in every mode.
        self.lift = nn.Conv3d(3, config.fno_width, 1)
        self.layers = nn.ModuleList(
            [
                ControlledLayer(
                    config.fno_width, config.q_cutoff, config.spectral_chunk_size
                )
                for _ in range(config.fno_layers)
            ]
        )
        self.readout = nn.Conv3d(config.fno_width, 1, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        for layer in self.layers:
            layer.configure(mode, activation)

    def forward(self, channels, cell):
        x = (channels - self.input_mean[:, None, None, None]) / self.input_scale[
            :, None, None, None
        ]
        x = self.lift(x[None])
        for layer in self.layers:
            x = layer(x, cell)
        return self.readout(x)[0, 0]


def channels(local, density, hxc):
    """A common fixed transform; scaling is fitted on training inputs only."""
    return torch.stack((local, density.clamp_min(0).pow(1 / 3), hxc))


def input_statistics(samples):
    count = sum(np.prod(s.shape) for s in samples)
    total = torch.zeros(3, dtype=torch.float64, device=samples[0].cell.device)
    squares = torch.zeros_like(total)
    for s in samples:
        x = channels(s.frozen_local, s.density, s.hxc).double().flatten(1)
        total += x.sum(1)
        squares += x.square().sum(1)
    mean = total / count
    scale = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


class ControlledGate(nn.Module):
    def __init__(self, config, mode, local_state, mean, scale, *, phase):
        super().__init__()
        if phase not in ("frozen", "joint"):
            raise ValueError("Unknown training phase")
        self.config, self.mode, self.phase = config, mode, phase
        self.local = LocalMACEField(config)
        self.local.load_state_dict(local_state)
        if phase == "frozen":
            self.local.requires_grad_(False)
        self.correction = ControlledCorrection(config, mode, mean, scale)

    def forward(self, sample):
        if self.phase == "frozen":
            local = sample.frozen_local
        else:
            graph = (
                sample.cpu_graph
                if next(self.local.parameters()).device.type == "cpu"
                else sample.graph
            )
            coefficients, offset = self.local.atomic_coefficients(graph)
            device = sample.dense_probes.device
            local = (
                torch.einsum(
                    "parl,arl->p", sample.dense_probes, coefficients.to(device)
                )
                + offset.to(device)
            ).reshape(sample.shape)
        field = local + self.correction(
            channels(local, sample.density, sample.hxc), sample.cell
        )
        return torch.exp(0.1 * torch.tanh(field))


def prepare_device(sample, device, reference_local):
    """Small-cell diagnostic: cache a bounded dense probe basis on the GPU.

    probe_basis enforces its existing 256-MiB per-structure limit. Production
    large-cell models should retain the streaming probe decoder instead.
    """
    inputs = FieldInputs(
        sample.data["numbers"],
        sample.data["positions"],
        sample.cell.numpy(),
        sample.density.numpy(),
        sample.hxc.numpy(),
    )
    sample.dense_probes = probe_basis(inputs, sample.shape, reference_local.config).to(
        device
    )
    sample.cpu_graph = sample.graph
    sample.graph = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.graph.items()
    }
    for key in ("psi", "target", "cell", "density", "hxc"):
        setattr(sample, key, getattr(sample, key).to(device))
    sample.op.orbitals = sample.op.orbitals.to(device)
    sample.op.masks = sample.op.masks.to(device)
    sample.op.kernels = [k.to(device) for k in sample.op.kernels]
    sample.op.recompute = False
    with torch.no_grad():
        sample.frozen_local = reference_local(
            sample.graph, sample.dense_probes
        ).reshape(sample.shape)
    return sample


def common_state(model):
    """Parameters whose initialization must be identical across correction modes."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("local.", "correction.lift.", "correction.readout."))
        or ".pointwise." in k
    }
