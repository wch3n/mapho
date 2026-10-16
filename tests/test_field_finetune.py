"""Joint gradient flow, frozen controls, and checkpoint provenance."""

import hashlib
import json

import numpy as np
import pytest
import torch

pytest.importorskip("mace")

from mapho.field_model import (
    FieldModelConfig,
    LocalMACEField,
    ResidualFNO,
    load_field_models,
    mace_graph,
)
from mapho.field_train import CoupledField, evaluate, finetune, train
from mapho.probes import PeriodicProbeGrid
from test_field_memory import make_dataset
from test_field_model import inputs


def config():
    return FieldModelConfig(
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


def test_joint_loss_reaches_mace_through_fno_input_and_direct_field():
    torch.manual_seed(11)
    c = config()
    i = inputs((4, 4, 4))
    local, fno = LocalMACEField(c), ResidualFNO(c)
    model = CoupledField(local, fno)
    item = {
        "graph": mace_graph(i, c),
        "probes": PeriodicProbeGrid(i, (4, 4, 4), c),
        "cell": torch.tensor(i.cell, dtype=torch.float32),
        "reference": torch.randn(2, 4, 4, 4),
        "target": torch.randn(4, 4, 4),
    }
    backbone = [p for p in local.backbone.parameters() if p.requires_grad]
    field = local(item["graph"], item["probes"]).reshape(4, 4, 4)
    correction = fno(torch.cat([field[None], item["reference"]]), item["cell"])
    through_fno = torch.autograd.grad(correction.square().mean(), backbone)
    assert any(g.abs().max() > 0 for g in through_fno if g.numel())

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = (model(item) - item["target"]).square().mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().max() > 0 for p in backbone if p.numel()
    )
    assert all(p.grad is None for p in local.backbone.readouts.parameters())

    # Finite differences cover both the direct field and its input to the FNO.
    parameter = local.coefficients.weight
    index = int(parameter.grad.abs().argmax())
    analytic = float(parameter.grad.flatten()[index])
    original = float(parameter.detach().flatten()[index])
    values = []
    with torch.no_grad():
        for delta in (1e-3, -1e-3):
            parameter.flatten()[index] = original + delta
            values.append(float((model(item) - item["target"]).square().mean()))
        parameter.flatten()[index] = original
    assert analytic == pytest.approx((values[0] - values[1]) / 2e-3, rel=0.02, abs=2e-4)
    optimizer.step()
    for prefix in ("local.backbone.interactions.", "local.coefficients.", "fno."):
        assert any(
            not torch.equal(p, before[name])
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        )


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "data"
    make_dataset(root)
    train(root, tmp_path / "staged", config(), local_epochs=1, fno_epochs=1)
    return root, tmp_path / "staged" / "model.pt"


def test_finetuning_modes_preserve_parent_scales_and_frozen_control(staged, tmp_path):
    root, checkpoint = staged
    parent_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parent_local, parent_fno, _ = load_field_models(checkpoint)
    starts = []
    for mode in ("fno", "joint"):
        out = tmp_path / mode
        local, fno = finetune(root, checkpoint, out, mode=mode, epochs=2, seed=13)
        history = json.loads((out / "history.json").read_text())["finetune"]
        metadata = json.loads((out / "runtime.json").read_text())
        starts.append(history["history"][0]["validation_rmse_eV"])
        assert history["best_validation_rmse_eV"] <= starts[-1]
        assert metadata["fine_tuning"]["parent_checkpoint_sha256"] == parent_hash
        assert metadata["fine_tuning"]["optimizer_steps"] == 4
        assert (
            metadata["parameter_changes"]["unused_energy_readouts"]["changed_tensors"]
            == 0
        )
        assert not any(p.requires_grad for p in local.backbone.readouts.parameters())
        assert any(p.requires_grad for p in local.parameters()) == (mode == "joint")
        torch.testing.assert_close(
            fno.input_scale, parent_fno.input_scale, rtol=0, atol=0
        )
        torch.testing.assert_close(
            fno.output_scale, parent_fno.output_scale, rtol=0, atol=0
        )
        assert (out / "fno_cache").exists() == (mode == "fno")
        if mode == "fno":
            for key, value in local.state_dict().items():
                torch.testing.assert_close(
                    value, parent_local.state_dict()[key], rtol=0, atol=0
                )
            assert metadata["parameter_changes"]["local"]["changed_tensors"] == 0
        report = evaluate(
            root, out / "model.pt", out / "evaluation", models=["local_fno"]
        )
        assert set(report["mean"]) == {"local_fno"}
        assert np.isfinite(report["mean"]["local_fno"]["potential_rmse_eV"])
    assert starts[0] == pytest.approx(starts[1], rel=2e-6, abs=1e-7)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == parent_hash


def test_finetuning_rejects_changed_manifest_before_writing(staged, tmp_path):
    root, checkpoint = staged
    manifest = root / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="split differs"):
        finetune(root, checkpoint, tmp_path / "bad", epochs=1)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="unknown"),
        dict(epochs=0),
        dict(local_lr=-1),
        dict(fno_lr=float("nan")),
    ],
)
def test_invalid_finetuning_arguments_fail_before_io(kwargs):
    with pytest.raises(ValueError):
        finetune("missing", "missing", "unused", **kwargs)
