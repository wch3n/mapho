"""Numerical and storage regressions for bounded field evaluation."""

import copy
import gc
import json
import weakref

import numpy as np
import pytest
import torch

pytest.importorskip("mace")

from mapho.field_model import (
    FieldModelConfig,
    LocalMACEField,
    MetricSpectralLayer,
    probe_basis,
    mace_graph,
    FieldInputs,
)
from mapho.probes import PeriodicProbeGrid
from test_field_model import inputs


@pytest.mark.parametrize(
    "shape,chunks", [((7, 6, 5), (13, 7)), ((4, 5, 6), (1000, 10000))]
)
def test_sparse_probes_match_dense_values_and_coefficient_gradients(shape, chunks):
    config = FieldModelConfig(
        probe_chunk_size=chunks[0], probe_edge_chunk_size=chunks[1]
    )
    i = inputs(shape)
    i.positions[0] += i.cell[0] * 2 - i.cell[1]
    dense = probe_basis(i, shape, config, torch.float64)
    grid = PeriodicProbeGrid(i, shape, config)
    coefficients = torch.randn(2, 12, 9, dtype=torch.float64, requires_grad=True)
    expected = torch.einsum("parl,arl->p", dense, coefficients)
    actual = grid.contract(coefficients)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    weight = torch.randn_like(actual)
    a = torch.autograd.grad((actual * weight).sum(), coefficients)[0]
    b = torch.autograd.grad((expected * weight).sum(), coefficients)[0]
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    assert all(len(p) <= chunks[1] for p, _, _ in grid.edges())


def test_probe_backward_saves_no_edge_bases_and_reaches_mace():
    c = FieldModelConfig(
        channels=2,
        radial_count=4,
        lmax=1,
        correlation=2,
        probe_chunk_size=8,
        probe_edge_chunk_size=5,
    )
    i = inputs((4, 4, 4))
    m = LocalMACEField(c)
    coeff, offset = m.atomic_coefficients(mace_graph(i, c))
    sparse = PeriodicProbeGrid(i, (4, 4, 4), c)
    shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda x: shapes.append(tuple(x.shape)) or x, lambda x: x
    ):
        prediction = sparse.contract(coeff)
    assert shapes == []
    dense = torch.einsum("parl,arl->p", probe_basis(i, (4, 4, 4), c), coeff)
    weight = torch.randn_like(dense)
    parameters = [p for p in m.parameters() if p.requires_grad]
    expected = torch.autograd.grad(
        ((dense + offset) * weight).sum(), parameters, retain_graph=True
    )
    actual = torch.autograd.grad(((prediction + offset) * weight).sum(), parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    assert any(g.abs().max() > 0 for g in actual)


def test_probe_empty_neighbors_and_multiple_images():
    c = FieldModelConfig(
        lmax=0,
        radial_count=4,
        probe_cutoff=3,
        probe_chunk_size=3,
        probe_edge_chunk_size=2,
    )
    i = FieldInputs(
        np.array([14]),
        np.array([[0.0, 0.0, 0.0]]),
        np.eye(3) * 2,
        np.ones((4, 4, 4)),
        np.ones((4, 4, 4)),
    )
    g = PeriodicProbeGrid(i, (4, 4, 4), c)
    coeff = torch.randn(1, 4, 1, requires_grad=True)
    torch.testing.assert_close(
        g.contract(coeff),
        torch.einsum("parl,arl->p", probe_basis(i, g.shape, c), coeff),
    )
    i.cell *= 10
    i.positions[:] = 10
    empty = PeriodicProbeGrid(i, (1, 1, 1), c).contract(coeff)
    assert empty.item() == 0
    empty.sum().backward()
    assert torch.count_nonzero(coeff.grad) == 0


def dense_spectral(layer, x, cell):
    """Original full-grid formula, independent of the optimized forward path."""
    shape = x.shape[-3:]
    axes = [torch.fft.fftfreq(n, dtype=x.dtype) * n for n in shape]
    modes = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    q = 2 * torch.pi * torch.einsum("ab,ijkb->ijka", torch.linalg.inv(cell), modes)
    q2 = q.square().sum(-1)
    weights = layer.radial(torch.log1p(q2)[..., None]).reshape(
        *shape, layer.width, layer.width
    )
    support = torch.ones(shape, dtype=torch.bool)
    for axis, n in enumerate(shape):
        if n % 2 == 0:
            support &= modes[..., axis] != -n // 2
    weights = (
        weights
        * ((1 - q2 / layer.q_cutoff**2).clamp_min(0).square() * support)[
            ..., None, None
        ]
        / layer.width**0.5
    )
    spectrum = torch.fft.fftn(x, dim=(-3, -2, -1))
    out = torch.einsum("bcijk,ijkcd->bdijk", spectrum, weights.to(spectrum.dtype))
    return torch.nn.functional.silu(
        torch.fft.ifftn(out, dim=(-3, -2, -1)).real + layer.pointwise(x)
    )


@pytest.mark.parametrize("chunk", [1, 17, 10000])
def test_chunked_spectral_values_and_all_gradients_match_dense(chunk):
    layer = MetricSpectralLayer(3, 4.0, chunk_size=chunk).double()
    x = torch.randn(2, 3, 6, 7, 8, dtype=torch.float64, requires_grad=True)
    cell = torch.tensor(
        [[3.0, 0.8, 0.1], [0.0, 4.0, 0.3], [0.4, 0.0, 5.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = dense_spectral(layer, x, cell)
    actual = layer(x, cell)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    weight = torch.randn_like(actual)
    variables = [x, cell, *layer.parameters()]
    a = torch.autograd.grad((actual * weight).sum(), variables)
    b = torch.autograd.grad((expected * weight).sum(), variables)
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, atol=1e-10, rtol=1e-9)


def test_spectral_radial_evaluation_is_bounded_and_skips_inactive_modes():
    layer = MetricSpectralLayer(4, 3.0, chunk_size=7)
    seen = []
    handle = layer.radial.register_forward_pre_hook(
        lambda m, args: seen.append(args[0].shape[0])
    )
    with torch.no_grad():
        layer(torch.randn(1, 4, 16, 16, 16), torch.eye(3) * 4)
    handle.remove()
    assert seen and max(seen) <= 7
    assert sum(seen) < 16**3 // 10


def make_dataset(root):
    from test_fields import example

    root.mkdir()
    splits = {"train": ["a.npz", "b.npz"], "validation": ["c.npz"], "test": ["d.npz"]}
    for n, name in enumerate(sum(splits.values(), [])):
        s = example()
        s.numbers[:] = 14
        s.positions[0, 0] += 0.07 * n
        s.metadata.update(functional="PBE", qe_config={"nbnd": 2})
        s.save(root / name)
    (root / "manifest.json").write_text(
        json.dumps({"schema": "mlham.field_dataset.v1", "splits": splits})
    )
    (root / "complete.json").write_text('{"count":4}')


def test_dataset_index_and_preparation_do_not_keep_samples(tmp_path, monkeypatch):
    from mapho import field_train as ft

    root = tmp_path / "data"
    make_dataset(root)
    original = ft.FieldSample.load
    references = []

    def track(path):
        sample = original(path)
        references.append(weakref.ref(sample))
        return sample

    monkeypatch.setattr(ft.FieldSample, "load", track)
    data, _, _ = ft.load_dataset(root)
    gc.collect()
    assert not any(r() is not None for r in references)
    prepared = ft._prepare(data["train"], FieldModelConfig(grid_size=4))
    item = prepared[0]
    gc.collect()
    assert not any(r() is not None for r in references)
    assert "basis" not in item and isinstance(item["probes"], PeriodicProbeGrid)
    assert set(vars(prepared)) == {"entries", "config"}


def test_streamed_training_checkpoint_and_disk_residuals(tmp_path):
    from mapho.field_train import train, evaluate, ResidualFiles

    root = tmp_path / "data"
    make_dataset(root)
    c = FieldModelConfig(
        channels=2,
        radial_count=4,
        lmax=1,
        correlation=2,
        grid_size=4,
        fno_width=2,
        fno_layers=1,
        probe_chunk_size=13,
        probe_edge_chunk_size=17,
        spectral_chunk_size=3,
    )
    local, fno = train(root, tmp_path / "fit", c, local_epochs=1, fno_epochs=1)
    files = sorted((tmp_path / "fit" / "fno_cache" / "train").glob("*.npz"))
    assert len(files) == 2
    assert set(ResidualFiles(files)[0]) == {"channels", "target", "cell"}
    assert not any(p.requires_grad for p in local.parameters())
    report = evaluate(root, tmp_path / "fit" / "model.pt", tmp_path / "evaluation")
    assert report["count"] == 1
    assert np.isfinite(report["mean"]["local_fno"]["potential_rmse_eV"])
