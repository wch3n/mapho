"""Exercise YAML validation, CLI precedence, real fitting, and provenance."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from mapho.commands.vasp import main
from mapho.experiments.vasp.configured_training import load_config, run
from mapho.vasp import train as training
from mapho.vasp.paw import save_json, sha


def recipe(path, **changes):
    data = dict(dataset="data", output="training")
    data.update(changes)
    path.write_text(yaml.safe_dump(data))
    return path


def test_paths_defaults_and_cli_overrides_without_dataset_access(
    tmp_path, monkeypatch, capsys
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = recipe(
        config_dir / "train.yaml", training=dict(epochs=100, backbone_lr="3e-4")
    )
    monkeypatch.chdir(tmp_path)
    main(
        [
            "train",
            "run",
            "--config",
            str(path),
            "--dry-run",
            "--epochs",
            "7",
            "--device",
            "cpu",
            "--output",
            "cli_output",
            "--mode",
            "local",
        ]
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["training"]["epochs"] == 7
    assert resolved["training"]["backbone_lr"] == 3e-4
    assert resolved["training"]["device"] == "cpu"
    assert resolved["mode"] == "local" and resolved["correction_activation"] == "gelu"
    assert resolved["dataset"] == str(config_dir / "data")
    assert resolved["atomic_reference"] == str(config_dir / "data/atomic")
    assert resolved["output"] == str(tmp_path / "cli_output")
    assert resolved["source_config_sha256"] == sha(path)
    assert not (tmp_path / "cli_output").exists()
    assert not (config_dir / "data").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"training": {"epoch": 4}},
        {"training": {"epochs": True}},
        {"training": {"epochs": 0}},
        {"training": {"seed": -1}},
        {"training": {"head_lr": float("nan")}},
        {"training": {"device": "gpu"}},
        {"training": {"scheduler_factor": 1}},
        {"model": {"atomic_numbers": [14]}},
        {"model": {"channels": 1.5}},
        {"model": {"grid_shape": [4, 4]}},
        {"model": {"probe_cutoff": float("inf")}},
        {"model": {"lmax": 3}},
        {"sampling": {"point_count": 0}},
        {"sampling": {"seed": -1}},
        {"model": None},
        {"sampling": None},
        {"training": None},
        {"version": 2},
        {"target": "hse"},
        {"target": "pbe0_components", "mode": "none"},
        {"target": "pbe0_components", "correction_activation": "silu"},
    ],
)
def test_invalid_configuration_fails_before_creating_output(tmp_path, changes):
    path = recipe(tmp_path / "train.yaml", **changes)
    with pytest.raises(ValueError):
        run(path)
    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize(
    "text",
    [
        "dataset: data\noutput: a\noutput: b\n",
        "dataset: data\noutput: a\ntraining: {epochs: 1, epochs: 2}\n",
        "!!python/object/apply:builtins.str [bad]\n",
    ],
)
def test_duplicate_keys_and_executable_yaml_are_rejected(tmp_path, text):
    path = tmp_path / "train.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_pbe0_audit_remains_mandatory(tmp_path, monkeypatch):
    from mapho.experiments.vasp import pbe0_learning

    path = recipe(tmp_path / "train.yaml", target="pbe0_components")
    monkeypatch.setattr(pbe0_learning, "audit", lambda root: dict(passed=False))
    with pytest.raises(ValueError, match="must pass before training"):
        run(path)
    assert not (tmp_path / "training").exists()


def test_cli_replaces_values_before_numeric_and_cross_field_validation(tmp_path):
    path = recipe(tmp_path / "train.yaml", training=dict(epochs=0, backbone_lr=1e-6))
    config = load_config(path, dict(epochs=2, backbone_lr=3e-4))
    assert config["training"]["epochs"] == 2
    assert config["training"]["backbone_lr"] == 3e-4


def synthetic_dataset(root):
    """Tiny nonphysical scalar PAW fixture; never passed to VASP."""
    root.mkdir()
    atomic = root / "atomic"
    atomic.mkdir()
    splits = {}
    shape = (4, 4, 6)
    x = np.arange(np.prod(shape)).reshape(shape)
    baseline = 0.2 * np.cos(x)
    for i, split in enumerate(("train", "validation", "test")):
        name = f"sample_{i:03d}"
        target = root / name
        reference = atomic / name
        target.mkdir()
        reference.mkdir()
        (target / "POSCAR").write_text(f"Synthetic distinct geometry {i}")
        splits[split] = [f"{name}/operator.h5"]
        for folder, offset in ((reference, 0.0), (target, 1.0)):
            with h5py.File(folder / "operator.h5", "w") as f:
                f.attrs["schema"] = "mapho.vasp_paw_operator.v1"
                f.attrs["metadata_json"] = json.dumps(
                    {"reference_manifest": {"settings": {"ICHARG": 12}}}
                )
                f["cell_A"] = np.diag([6.0, 6.0, 7.0])
                f["positions_A"] = [[0.2 + i * 0.01, 0.3, 0.4], [2.0, 1.4, 1.5]]
                f["atomic_numbers"] = [14, 14]
                f["paw/types"] = [0, 0]
                f["paw/d"] = np.broadcast_to(
                    np.eye(8) * (0.1 + 0.03 * offset), (2, 8, 8)
                )
                f["paw/q"] = np.zeros((2, 8, 8))
                f["paw"].attrs["channels_json"] = "[[0, 0, 1, 1]]"
                f["potential_eV"] = baseline + offset * (0.03 + 0.02 * np.sin(x))
    save_json(root / "manifest.json", dict(splits=splits))
    save_json(root / "data_audit.json", dict(passed=True))
    save_json(atomic / "completion.json", dict(complete=True))
    save_json(
        atomic / "manifest.json",
        dict(dataset_manifest_sha256=sha(root / "manifest.json")),
    )


@pytest.mark.parametrize("mode", ["local", "full"])
def test_yaml_runs_actual_training_and_preserves_effective_settings(tmp_path, mode):
    synthetic_dataset(tmp_path / "data")
    path = recipe(
        tmp_path / "train.yaml",
        mode=mode,
        model=dict(
            channels=2,
            lmax=1,
            correlation=1,
            radial_count=2,
            grid_size=8,
            grid_shape=[4, 4, 6],
            fno_width=2,
            fno_layers=1,
        ),
        sampling=dict(point_count=12, seed=920),
        training=dict(
            epochs=99,
            device="cpu",
            seed=47,
            backbone_lr=2e-4,
            head_lr=8e-4,
            validation_interval=1,
        ),
    )
    root = tmp_path / "training"
    args = ["train", "run", "--config", str(path), "--epochs", "2"]
    if mode == "local":
        main(args + ["--prepare-only"])
        # Existing commands must honor the saved CPU device and epoch count.
        main(["train", "model", "train", str(root), mode])
        main(["train", "model", "evaluate", str(root), mode])
    else:
        main(args)
    manifest = json.loads((root / "manifest.json").read_text())
    resolved = json.loads((root / "config.resolved.json").read_text())
    assert (root / "config.input.yaml").read_bytes() == path.read_bytes()
    assert resolved["training"]["epochs"] == manifest["training"]["epochs"] == 2
    assert manifest["config"]["channels"] == 2
    assert manifest["config"]["grid_shape"] == [4, 4, 6]
    with np.load(root / "cache/sample_000.npz") as cache:
        assert cache["baseline"].shape == (4, 4, 6)
        assert len(cache["fractions"]) == 12
    history = json.loads((root / mode / "history.json").read_text())
    assert history[1]["learning_rates"] == [2e-4, 8e-4]
    report = json.loads((root / mode / "training.json").read_text())
    assert report["updates"] == 2 and report["seed"] == 47
    assert report["parameter_changes"]["local.backbone"]["changed_tensors"] > 0
    if mode == "full":
        assert report["parameter_changes"]["correction"]["changed_tensors"] > 0
    checkpoint = torch.load(root / mode / "model.pt", weights_only=False)
    assert checkpoint["training_options"] == report["training_options"]
    assert checkpoint["description"]["correction_activation"] == "gelu"
    model = training.load_model(root, mode, "cpu")
    field, d = training.predict(model, manifest["entries"]["test"][0], "cpu")
    assert field.shape == (4, 4, 6) and np.isfinite(d).all()
    assert (root / mode / "test_fields.json").is_file()
    with pytest.raises(FileExistsError):
        run(path)
    with (root / "config.input.yaml").open("a") as f:
        f.write("\n# modified\n")
    with pytest.raises(ValueError, match="configuration changed"):
        training.load_model(root, mode, "cpu")


def test_prepare_only_writes_no_checkpoint_and_preserves_existing_cli(tmp_path, capsys):
    synthetic_dataset(tmp_path / "data")
    path = recipe(
        tmp_path / "train.yaml",
        model=dict(grid_size=4),
        sampling=dict(point_count=8),
        training=dict(device="cpu", epochs=2),
    )
    result = run(path, prepare_only=True)
    assert result["prepared"] and not result["trained"]
    assert not (tmp_path / "training/full").exists()
    with pytest.raises(SystemExit) as error:
        main(["train", "model", "train", "--help"])
    assert error.value.code == 0


def test_shipped_recipes_resolve():
    examples = Path(__file__).resolve().parents[1] / "examples/training"
    for path in examples.glob("*.yaml"):
        config = load_config(path)
        assert config["correction_activation"] == "gelu"
