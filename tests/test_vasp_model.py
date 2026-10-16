import copy
import numpy as np
import pytest
import torch
from ase.build import bulk
from e3nn import o3

from mapho.field_model import FieldModelConfig, mace_graph, probe_basis
from mapho.vasp.model import (
    PAWModel,
    atomic_inputs,
    native_sp_order,
    point_basis,
    periodic_interpolate,
)


def test_gelu_checkpoint_roundtrip_and_contract(tmp_path):
    import json
    from mapho.vasp.paw import sha
    from mapho.vasp.train import load_model

    config = FieldModelConfig(
        channels=2,
        radial_count=4,
        correlation=2,
        grid_size=6,
        fno_width=2,
        fno_layers=2,
    )
    models = {}
    for activation in ("silu", "gelu"):
        torch.manual_seed(31)
        models[activation] = PAWModel(
            config, mode="full", correction_activation=activation
        ).eval()
    for key, value in models["silu"].state_dict().items():
        torch.testing.assert_close(
            models["gelu"].state_dict()[key], value, rtol=0, atol=0
        )
    model = models["gelu"]
    with torch.no_grad():
        model.correction.readout.weight.fill_(0.1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"correction_activation": "gelu"}))
    (tmp_path / "full").mkdir()
    checkpoint = dict(
        state_dict=model.state_dict(),
        description=model.description(),
        training_manifest_sha256=sha(manifest),
    )
    torch.save(checkpoint, tmp_path / "full/model.pt")
    restored = load_model(tmp_path, "full", "cpu")
    assert all(layer.activation_name == "gelu" for layer in restored.correction.layers)
    data = item(sample(), config)
    a, da = model(data)
    b, db = restored(data)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(da, db, rtol=0, atol=0)
    checkpoint["description"]["correction_activation"] = "silu"
    torch.save(checkpoint, tmp_path / "full/model.pt")
    with pytest.raises(ValueError, match="activation/mode"):
        load_model(tmp_path, "full", "cpu")


def sample():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1] += [0.04, -0.02, 0.03]
    atoms.set_cell(
        atoms.cell.array @ np.array([[1, 0.02, 0], [0, 0.98, 0.01], [0, 0, 1.01]]),
        scale_atoms=True,
    )
    return atomic_inputs(
        atoms.numbers, atoms.positions, atoms.cell.array, np.zeros((6, 6, 6))
    )


def item(inputs, config):
    return dict(
        graph=mace_graph(inputs, config),
        probes=probe_basis(inputs, (6, 6, 6), config),
        shape=(6, 6, 6),
        baseline=torch.tensor(inputs.reference_hxc, dtype=torch.float32),
        envelope=torch.tensor(inputs.reference_density, dtype=torch.float32),
        cell=torch.tensor(inputs.cell, dtype=torch.float32),
    )


def test_native_p_convention_matches_cartesian_harmonics():
    vectors = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]], dtype=torch.float64)
    vectors /= vectors.norm(dim=-1, keepdim=True)
    sh = o3.spherical_harmonics(
        [0, 1], vectors, normalize=True, normalization="integral"
    )
    native = sh[:, native_sp_order([0, 1])]
    expected = torch.cat(
        [
            torch.ones((2, 1), dtype=torch.float64) / np.sqrt(4 * np.pi),
            vectors[:, [1, 2, 0]] * np.sqrt(3 / (4 * np.pi)),
        ],
        dim=-1,
    )
    torch.testing.assert_close(native, expected)
    with pytest.raises(ValueError):
        native_sp_order([2])


def test_symmetric_readout_is_complete_and_equivariant_with_gradients():
    torch.manual_seed(4)
    config = FieldModelConfig(
        channels=3,
        radial_count=4,
        correlation=2,
        grid_size=6,
        fno_width=3,
        fno_layers=1,
    )
    model = PAWModel(config, mode="full")
    # Nonzero heads are required to test covariance and backbone gradients.
    for name, p in model.named_parameters():
        if name.startswith(
            ("local.coefficients", "local.offset", "onsite_", "correction.readout")
        ):
            torch.nn.init.normal_(p, std=0.04)
    basis = model.matrix_basis.flatten(1)
    torch.testing.assert_close(basis @ basis.T, torch.eye(36), atol=3e-6, rtol=3e-6)
    a = sample()
    b = copy.deepcopy(a)
    rotation = o3.rand_matrix()
    b.cell = a.cell @ rotation.numpy().T
    b.positions = a.positions[::-1].copy() @ rotation.numpy().T
    b.numbers = a.numbers[::-1].copy()
    fa, da = model(item(a, config))
    fb, db = model(item(b, config))
    rep = o3.Irreps("0e+0e+1o+1o").D_from_matrix(rotation)
    order = native_sp_order([0, 0, 1, 1])
    native = rep[order][:, order]
    torch.testing.assert_close(fa, fb, atol=3e-6, rtol=3e-4)
    torch.testing.assert_close(db, native @ da.flip(0) @ native.T, atol=3e-6, rtol=3e-4)
    torch.testing.assert_close(da, da.transpose(-1, -2))
    (fa.square().mean() + da.square().mean()).backward()
    for prefix in (
        "local.backbone.products",
        "onsite_linear",
        "onsite_product",
        "correction",
    ):
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        ]
        assert any(g is not None and g.abs().max() > 0 for g in grads), prefix
        assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_atomic_inputs_are_geometry_only_and_zero_head_returns_baseline_correction():
    config = FieldModelConfig(
        channels=2,
        radial_count=4,
        correlation=2,
        grid_size=6,
        fno_width=2,
        fno_layers=1,
    )
    inputs = sample()
    model = PAWModel(config, mode="none")
    field, d = model(item(inputs, config))
    assert torch.count_nonzero(field) == 0 and torch.count_nonzero(d) == 0
    assert inputs.reference_density.mean() * np.linalg.det(
        inputs.cell
    ) == pytest.approx(2.0)


def test_native_point_decoder_matches_grid_and_interpolation_is_periodic():
    inputs = sample()
    config = FieldModelConfig(radial_count=4, grid_size=6)
    ids = np.array([0, 3, 38, 215])
    fractions = np.stack(np.unravel_index(ids, (6, 6, 6)), -1) / 6
    torch.testing.assert_close(
        point_basis(inputs, fractions, config),
        probe_basis(inputs, (6, 6, 6), config)[ids],
        atol=1e-6,
        rtol=1e-6,
    )
    field = torch.randn(6, 6, 6, requires_grad=True)
    points = torch.tensor(fractions, dtype=torch.float32)
    selected = periodic_interpolate(field, points)
    torch.testing.assert_close(selected, field.flatten()[ids])
    torch.testing.assert_close(
        selected, periodic_interpolate(field, points + 1), atol=3e-6, rtol=3e-6
    )
    selected.sum().backward()
    assert torch.isfinite(field.grad).all()


def test_native_prediction_needs_no_target_operator(tmp_path):
    import h5py
    from mapho.vasp.train import predict
    from mapho.file_utils import sha

    inputs = sample()
    path = tmp_path / "atomic.h5"
    baseline = np.random.default_rng(2).normal(size=(6, 6, 6))
    with h5py.File(path, "w") as f:
        f["potential_eV"] = baseline
        f["paw/d"] = np.zeros((2, 8, 8))
        f["atomic_numbers"] = inputs.numbers
        f["positions_A"] = inputs.positions
        f["cell_A"] = inputs.cell
    config = FieldModelConfig(
        channels=2,
        radial_count=4,
        correlation=2,
        grid_size=6,
        fno_width=2,
        fno_layers=1,
    )
    model = PAWModel(config, mode="full").eval()
    entry = dict(
        baseline=str(path),
        baseline_sha256=sha(path),
        target=str(tmp_path / "does_not_exist.h5"),
    )
    seen = []
    hook = model.correction.register_forward_pre_hook(
        lambda *_: seen.append(
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        )
    )
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        potential, d = predict(model, entry, "cpu")
        assert seen == [(False, False)]  # Inference must match the training precision.
    finally:
        hook.remove()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    np.testing.assert_array_equal(potential, baseline)
    np.testing.assert_array_equal(d, np.zeros((2, 8, 8)))


def test_native_d_harmonics_match_cartesian_polynomials():
    from mapho.vasp.model import native_projector_transform

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        v = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0], [0.8, -0.2, -0.3]])
        v = v / v.norm(dim=-1, keepdim=True)
        x, y, z = v.T
        transform = native_projector_transform([0, 1, 2])
        sh = o3.spherical_harmonics(
            [0, 1, 2], v, normalize=True, normalization="integral"
        )
        factor = 1 / np.sqrt(4 * np.pi)
        expected = (
            torch.stack(
                [
                    torch.ones_like(x),
                    np.sqrt(3) * y,
                    np.sqrt(3) * z,
                    np.sqrt(3) * x,
                    np.sqrt(15) * x * y,
                    np.sqrt(15) * y * z,
                    np.sqrt(5) / 2 * (3 * z * z - 1),
                    np.sqrt(15) * x * z,
                    np.sqrt(15) / 2 * (x * x - y * y),
                ],
                -1,
            )
            * factor
        )
        torch.testing.assert_close(sh @ transform.T, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(
            transform @ transform.T, torch.eye(9), atol=1e-14, rtol=1e-14
        )
    finally:
        torch.set_default_dtype(old)


def test_two_species_spd_heads_rotation_permutation_padding_and_gradients(tmp_path):
    from mapho.vasp.model import native_projector_transform

    torch.manual_seed(4)
    config = FieldModelConfig(
        atomic_numbers=(16, 42), channels=2, radial_count=4, correlation=2, grid_size=6
    )
    channels = {16: [0, 1], 42: [0, 1, 2]}
    model = PAWModel(config, mode="local", species_channels=channels)
    for name, p in model.named_parameters():
        if name.startswith(("onsite_heads", "local.coefficients", "local.offset")):
            torch.nn.init.normal_(p, std=0.05)
    a = atomic_inputs(
        np.array([16, 42]),
        np.array([[0.7, 0.3, 0.5], [2.4, 1.2, 1.7]]),
        np.eye(3) * 6,
        np.zeros((6, 6, 6)),
    )
    rotation = o3.rand_matrix()
    b = copy.deepcopy(a)
    b.numbers = a.numbers[::-1].copy()
    b.positions = a.positions[::-1].copy() @ rotation.numpy().T
    b.cell = a.cell @ rotation.numpy().T
    fa, da = model(item(a, config))
    fb, db = model(item(b, config))
    torch.testing.assert_close(fa, fb, atol=5e-6, rtol=5e-4)
    for old_index, z in enumerate(a.numbers):
        ls = channels[int(z)]
        n = sum(2 * l + 1 for l in ls)
        t = native_projector_transform(ls)
        rep = o3.Irreps([(1, (l, (-1) ** l)) for l in ls]).D_from_matrix(rotation)
        native = t @ rep @ t.T
        torch.testing.assert_close(
            db[1 - old_index, :n, :n],
            native @ da[old_index, :n, :n] @ native.T,
            atol=5e-6,
            rtol=5e-4,
        )
        basis = model.onsite_heads[str(z)].matrix_basis.flatten(1)
        torch.testing.assert_close(
            basis @ basis.T, torch.eye(n * (n + 1) // 2), atol=3e-6, rtol=3e-6
        )
    inverted = copy.deepcopy(a)
    inverted.positions = -a.positions
    di = model.coefficients(mace_graph(inverted, config))[2]
    for index, z in enumerate(a.numbers):
        parity = torch.diag(
            torch.tensor(
                [(-1.0) ** l for l in channels[int(z)] for _ in range(2 * l + 1)]
            )
        )
        n = len(parity)
        torch.testing.assert_close(
            di[index, :n, :n], parity @ da[index, :n, :n] @ parity, atol=5e-6, rtol=5e-4
        )
    import json
    from mapho.vasp.train import load_model
    from mapho.vasp.paw import sha

    (tmp_path / "manifest.json").write_text(
        json.dumps({"correction_activation": "silu"})
    )
    (tmp_path / "local").mkdir()
    torch.save(
        dict(
            state_dict=model.state_dict(),
            description=model.description(),
            training_manifest_sha256=sha(tmp_path / "manifest.json"),
        ),
        tmp_path / "local/model.pt",
    )
    restored = load_model(tmp_path, "local", "cpu")
    torch.testing.assert_close(restored(item(a, config))[1], da, rtol=0, atol=0)
    assert (
        torch.count_nonzero(da[0, 4:]) == 0 and torch.count_nonzero(da[0, :, 4:]) == 0
    )
    (fa.square().mean() + da.square().mean()).backward()
    for prefix in ("local.backbone.products", "onsite_heads.16", "onsite_heads.42"):
        assert any(
            p.grad is not None and p.grad.abs().max() > 0
            for n, p in model.named_parameters()
            if n.startswith(prefix)
        )


def test_onsite_loss_excludes_native_padding():
    from mapho.vasp.train import onsite_mse

    pred = torch.ones(2, 9, 9, requires_grad=True)
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[0, 4:] = False
    mask[0, :, 4:] = False
    target = torch.zeros_like(pred)
    target[~mask] = 1.0e6
    loss = onsite_mse(pred, dict(target_d=target, d_mask=mask))
    assert loss.item() == 1.0
    loss.backward()
    assert torch.count_nonzero(pred.grad[~mask]) == 0
