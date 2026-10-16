import copy

import numpy as np
import pytest
import torch
from ase.build import bulk
from e3nn import o3

pytest.importorskip("mace")

from mapho.field_model import (
    FieldInputs,
    FieldModelConfig,
    LocalMACEField,
    ResidualFNO,
    MetricSpectralLayer,
    probe_basis,
    mace_graph,
    resize_field,
    predict_fields,
    save_field_models,
    load_field_models,
)
from mapho.probes import PeriodicProbeGrid


def inputs(shape=(6, 7, 8)):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_cell(
        atoms.cell.array @ np.array([[1.02, 0.04, 0], [0, 0.98, 0.02], [0, 0, 1.01]]),
        scale_atoms=True,
    )
    atoms.positions[1] += [0.04, -0.02, 0.01]
    return FieldInputs(
        atoms.numbers,
        atoms.positions,
        atoms.cell.array,
        np.ones(shape) * 0.2,
        np.ones(shape) * -3,
    )


@pytest.fixture
def model():
    return LocalMACEField(
        FieldModelConfig(
            channels=3,
            radial_count=5,
            correlation=2,
            grid_size=6,
            fno_width=3,
            fno_layers=1,
        )
    )


def test_local_rigid_rotation_and_atom_permutation(model):
    original = inputs()
    transformed = copy.deepcopy(original)
    q = o3.rand_matrix().numpy()
    transformed.cell = original.cell @ q.T
    transformed.positions = original.positions[::-1] @ q.T
    transformed.numbers = original.numbers[::-1].copy()
    shape = (6, 7, 8)
    a = model(
        mace_graph(original, model.config),
        PeriodicProbeGrid(original, shape, model.config),
    )
    b = model(
        mace_graph(transformed, model.config),
        PeriodicProbeGrid(transformed, shape, model.config),
    )
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-4)


def test_local_periodic_translation_and_manybody_gradients(model):
    original = inputs((6, 6, 6))
    translated = copy.deepcopy(original)
    translated.positions += original.cell[0] / 6 + original.cell[1] * 2
    shape = (6, 6, 6)
    a = model(
        mace_graph(original, model.config),
        PeriodicProbeGrid(original, shape, model.config),
    ).reshape(shape)
    b = model(
        mace_graph(translated, model.config),
        PeriodicProbeGrid(translated, shape, model.config),
    ).reshape(shape)
    torch.testing.assert_close(a.roll(1, 0), b, atol=3e-6, rtol=3e-4)
    a.square().mean().backward()
    gradients = [
        p.grad for n, p in model.named_parameters() if "backbone.products" in n
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)


def test_multiple_images_are_summed():
    c = FieldModelConfig(lmax=0, probe_cutoff=3, radial_count=4)
    i = FieldInputs(
        np.array([14]),
        np.array([[0.0, 0.0, 0.0]]),
        np.eye(3) * 2,
        np.ones((4, 4, 4)),
        np.ones((4, 4, 4)),
    )
    b = probe_basis(i, (4, 4, 4), c)
    # At the origin the basis includes the origin atom AND neighbours at 2 A.
    nearest_only = torch.exp(-0.5 * (torch.linspace(0, 3, 4)) ** 2)
    assert torch.all(b[0, 0, :, 0] > nearest_only)


def test_metric_fno_rotation_translation_and_cell_dependence():
    f = ResidualFNO(FieldModelConfig(fno_width=3, fno_layers=1))
    x = torch.randn(3, 8, 9, 10)
    cell = torch.tensor([[4.0, 0.3, 0.2], [0.4, 5.0, 0.0], [0.0, 0.1, 6.0]])
    q = o3.rand_matrix()
    a = f(x, cell)
    torch.testing.assert_close(f(x, cell @ q.T), a, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        f(x.roll((2, -1, 3), (1, 2, 3)), cell),
        a.roll((2, -1, 3), (0, 1, 2)),
        atol=1e-6,
        rtol=1e-5,
    )
    assert (f(x, cell * 1.2) - a).abs().max() > 1e-5
    a.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters()
    )


def test_fourier_resampling_offset_modes_and_nyquist():
    for shape in [(7, 8, 9), (8, 8, 8)]:
        f = 2 + np.cos(2 * np.pi * np.arange(shape[0]) / shape[0])[:, None, None]
        f = np.broadcast_to(f, shape)
        resized = resize_field(f, (12, 13, 14))
        expected = np.broadcast_to(
            2 + np.cos(2 * np.pi * np.arange(12) / 12)[:, None, None], resized.shape
        )
        np.testing.assert_allclose(resized, expected, atol=1e-13)
        np.testing.assert_allclose(resize_field(resized, shape), f, atol=1e-13)
    nyquist = np.broadcast_to((-1.0) ** np.arange(8)[:, None, None], (8, 8, 8))
    np.testing.assert_allclose(
        resize_field(resize_field(nyquist, (16, 16, 16)), (8, 8, 8)),
        nyquist,
        atol=1e-13,
    )


def test_label_independent_inputs_and_checkpoint_reload(model, tmp_path):
    from test_fields import example

    sample = example()
    sample.numbers[:] = 14
    i = FieldInputs.from_sample(sample)
    changed = copy.deepcopy(sample)
    changed.fields["hartree"] += 123
    changed.fields["xc"] *= 10
    changed.fields["density"] *= 0.7
    changed.eigenvalues += 1e4
    j = FieldInputs.from_sample(changed)
    f = ResidualFNO(model.config)
    a = predict_fields(model, f, i)
    np.testing.assert_array_equal(predict_fields(model, f, j), a)
    save_field_models(tmp_path / "model.pt", model, f, {"test": True})
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    assert state["schema"] == "mapho.field_model.v1"
    # Historical checkpoints retain their old tag but must give identical fields.
    state["schema"] = "mlham.field_model.v1"
    torch.save(state, tmp_path / "legacy.pt")
    old_local, old_fno, _ = load_field_models(tmp_path / "legacy.pt")
    np.testing.assert_array_equal(predict_fields(old_local, old_fno, i), a)
    local, fno, metadata = load_field_models(tmp_path / "model.pt")
    np.testing.assert_array_equal(predict_fields(local, fno, i), a)
    assert metadata == {"test": True}


def test_unknown_species_and_bad_config_rejected(model):
    i = inputs()
    i.numbers[:] = 6
    with pytest.raises(ValueError, match="element"):
        mace_graph(i, model.config)
    with pytest.raises(ValueError):
        FieldModelConfig(lmax=5).validate()


def test_band_metrics_preserve_absolute_offset():
    from mapho.field_train import electronic_metrics
    from test_fields import example

    sample = example()
    solution = {
        "eigenvalues_eV": sample.eigenvalues + 0.25,
        "density": sample.fields["density"],
    }
    metrics = electronic_metrics(solution, sample)
    assert metrics["band_mae_eV"] == 0.25
    assert metrics["band_aligned_mae_eV"] == 0
    assert metrics["sampled_gap_abs_error_eV"] == 0
    assert metrics["density_relative_l1"] == 0
