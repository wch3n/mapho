from types import SimpleNamespace
import pytest
import torch
from mapho.exchange_ablation import (
    ControlledCorrection,
    ControlledLayer,
    ControlledGate,
)
from mapho.field_model import (
    FieldModelConfig,
    MetricSpectralLayer,
    mace_graph,
    probe_basis,
)


@pytest.mark.parametrize("mode", ["none", "full"])
def test_gelu_changes_only_shared_layer_activation(mode):
    import copy

    models = {}
    for activation in ("silu", "gelu"):
        torch.manual_seed(31)
        models[activation] = ControlledCorrection(
            FieldModelConfig(fno_width=3, fno_layers=2), mode, activation=activation
        )
    for key, value in models["silu"].state_dict().items():
        torch.testing.assert_close(
            models["gelu"].state_dict()[key], value, rtol=0, atol=0
        )
    x = torch.randn(1, 3, 7, 8, 9, requires_grad=True)
    cell = torch.eye(3) * 4
    for silu, gelu in zip(models["silu"].layers, models["gelu"].layers):
        linear = copy.deepcopy(silu)
        linear.activate = lambda values: values
        preactivation = linear(x, cell)
        torch.testing.assert_close(
            silu(x, cell), torch.nn.functional.silu(preactivation), rtol=0, atol=0
        )
        actual = gelu(x, cell)
        torch.testing.assert_close(
            actual,
            torch.nn.functional.gelu(preactivation, approximate="none"),
            rtol=0,
            atol=0,
        )
        assert not torch.allclose(actual, silu(x, cell))
        grad = torch.autograd.grad(actual.sum(), x)[0]
        assert torch.isfinite(grad).all() and grad.abs().max() > 0
        if mode == "full":
            assert isinstance(gelu.radial[1], torch.nn.SiLU)
        else:
            grad = torch.autograd.grad(gelu(x, cell)[0, :, 0, 0, 0].sum(), x)[0]
            assert grad[:, :, 1:].count_nonzero() == 0
            assert grad[:, :, 0, 1:].count_nonzero() == 0
            assert grad[:, :, 0, 0, 1:].count_nonzero() == 0


def test_common_initialization_and_parameter_budget():
    c = FieldModelConfig(fno_width=16, fno_layers=2)
    models = {}
    for mode in ("none", "full", "local", "mean"):
        torch.manual_seed(31)
        models[mode] = ControlledCorrection(c, mode)
    for key, value in models["none"].state_dict().items():
        for m in models.values():
            torch.testing.assert_close(m.state_dict()[key], value, rtol=0, atol=0)
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    assert counts["none"] == 625
    assert counts["full"] == counts["mean"] == 9393
    assert abs(counts["full"] - counts["local"]) / counts["full"] < 0.003


def test_full_layer_reuses_original_and_none_is_spatially_pointwise():
    torch.manual_seed(7)
    original = MetricSpectralLayer(3, 6)
    full = ControlledLayer(3, 6)
    full.load_state_dict(original.state_dict())
    full.configure("full")
    x = torch.randn(1, 3, 7, 8, 9, requires_grad=True)
    cell = torch.eye(3) * 4
    torch.testing.assert_close(full(x, cell), original(x, cell), rtol=0, atol=0)
    for mode in ("none", "local"):
        layer = ControlledLayer(3, 6)
        layer.configure(mode)
        y = layer(x, cell)[0, :, 0, 0, 0].sum()
        grad = torch.autograd.grad(y, x)[0]
        assert grad[:, :, 1:].count_nonzero() == 0
        assert grad[:, :, 0, 1:].count_nonzero() == 0
        assert grad[:, :, 0, 0, 1:].count_nonzero() == 0


def test_mean_branch_equals_only_dc_fourier_contribution():
    layer = ControlledLayer(3, 6)
    layer.configure("mean")
    x = torch.randn(2, 3, 7, 8, 9)
    cell = torch.eye(3) * 4
    spectrum = torch.fft.fftn(x, dim=(-3, -2, -1))
    result = torch.zeros_like(spectrum)
    result[:, :, 0, 0, 0] = layer._mix(spectrum[:, :, 0, 0, 0, None], x.new_zeros(1))[
        :, :, 0
    ]
    expected = torch.nn.functional.silu(
        layer.pointwise(x) + torch.fft.ifftn(result, dim=(-3, -2, -1)).real
    )
    torch.testing.assert_close(layer(x, cell), expected, atol=2e-7, rtol=2e-6)


def test_all_controls_periodic_translation_and_gradients():
    for mode in ("none", "full", "local", "mean"):
        c = FieldModelConfig(fno_width=3, fno_layers=2)
        m = ControlledCorrection(c, mode)
        torch.nn.init.normal_(m.readout.weight, std=0.1)
        x = torch.randn(3, 7, 8, 9)
        cell = torch.tensor([[4.0, 0.2, 0], [0, 4.0, 0.1], [0.1, 0, 5.0]])
        y = m(x, cell)
        torch.testing.assert_close(
            m(x.roll(2, 1), cell), y.roll(2, 0), atol=1e-6, rtol=1e-5
        )
        y.square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
        )


def test_frozen_and_joint_gate_match_and_only_joint_updates_mace():
    from test_field_model import inputs
    from mapho.field_model import LocalMACEField

    c = FieldModelConfig(
        channels=3,
        radial_count=5,
        correlation=2,
        grid_size=6,
        fno_width=3,
        fno_layers=1,
    )
    data = inputs((6, 6, 6))
    graph = mace_graph(data, c)
    probes = probe_basis(data, data.reference_density.shape, c)
    reference = LocalMACEField(c)
    with torch.no_grad():
        local = reference(graph, probes).reshape(data.reference_density.shape)
    sample = SimpleNamespace(
        cpu_graph=graph,
        graph=graph,
        dense_probes=probes,
        frozen_local=local,
        shape=data.reference_density.shape,
        density=torch.tensor(data.reference_density).float(),
        hxc=torch.tensor(data.reference_hxc).float(),
        cell=torch.tensor(data.cell).float(),
    )
    for phase in ("frozen", "joint"):
        model = ControlledGate(
            c,
            "full",
            reference.state_dict(),
            torch.zeros(3),
            torch.ones(3),
            phase=phase,
        )
        gate = model(sample)
        torch.testing.assert_close(gate, torch.exp(0.1 * torch.tanh(local)))
        gate.square().mean().backward()
        gradients = [p.grad for p in model.local.parameters()]
        if phase == "frozen":
            assert all(g is None for g in gradients)
        else:
            assert any(g is not None and g.abs().sum() > 0 for g in gradients)
            assert all(g is None or torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("phase", ["frozen", "joint"])
def test_interrupted_training_resumes_identically(tmp_path, monkeypatch, phase):
    from dataclasses import dataclass
    from mapho import exchange_ablation_train as runner

    @dataclass
    class TinyConfig:
        width: int = 1

    class TinyGate(torch.nn.Module):
        def __init__(self, config, mode, state, mean, scale, *, phase):
            super().__init__()
            self.local = torch.nn.Sequential(torch.nn.Linear(1, 1))
            self.correction = torch.nn.Linear(1, 1)
            if phase == "frozen":
                self.local.requires_grad_(False)

        def forward(self, sample):
            return self.correction(self.local(sample))

    monkeypatch.setattr(runner, "ControlledGate", TinyGate)
    monkeypatch.setattr(runner, "action_mse", lambda s, g, k: (g - s).square().mean())
    args = SimpleNamespace(
        output=tmp_path / "continuous",
        device="cpu",
        local_device="cpu",
        smoke=True,
        max_epochs=4,
        min_epochs=4,
        patience_epochs=60,
    )
    args.output.mkdir()
    samples = [torch.tensor([[0.5]]), torch.tensor([[1.5]])]
    inputs = (
        phase,
        "none",
        31,
        TinyConfig(),
        {},
        torch.zeros(3),
        torch.ones(3),
        samples,
        samples,
        "test-protocol",
    )
    runner.train_one(args, *inputs)
    expected = torch.load(
        args.output / f"{phase}_none_seed31/model.pt", weights_only=True
    )
    args.output = tmp_path / "resumed"
    args.output.mkdir()
    save = runner.save_progress

    def interrupt(path, state):
        save(path, state)
        if state["epoch"] == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "save_progress", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train_one(args, *inputs)
    monkeypatch.setattr(runner, "save_progress", save)
    runner.train_one(args, *inputs)
    actual = torch.load(
        args.output / f"{phase}_none_seed31/model.pt", weights_only=True
    )
    for name, value in expected["state_dict"].items():
        torch.testing.assert_close(value, actual["state_dict"][name], rtol=0, atol=0)
    assert (
        actual["metadata"]["best_validation_rmse_meV"]
        == expected["metadata"]["best_validation_rmse_meV"]
    )
    assert actual["metadata"]["epochs"] == 4
