"""CLI dispatch stays separate from reusable inference and calculation workflows."""

import importlib
import json
import os
import subprocess
import sys

import pytest

from mapho.commands.vasp import COMMANDS, main


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_help_without_running_calculations(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_paw_dispatch_preserves_positional_order_and_shift(monkeypatch, capsys):
    from mapho.vasp import paw

    calls = []

    def compare(*args):
        calls.append(args)
        return {"passed": True}

    monkeypatch.setattr(paw, "compare", compare)
    main(["paw", "compare", "reference", "prediction", "--shift-eV", "0.25"])
    assert calls == [("reference", "prediction", 0.25)]
    assert json.loads(capsys.readouterr().out) == {"passed": True}


def test_silicon_dispatch_uses_workflow_namespace(monkeypatch, capsys):
    from mapho.workflows.vasp import silicon_dataset

    monkeypatch.setattr(silicon_dataset, "convergence", lambda root: {"root": root})
    main(["silicon-dataset", "convergence", "study"])
    assert json.loads(capsys.readouterr().out) == {"root": "study"}


def test_training_validation_calls_reusable_training_api(monkeypatch):
    from mapho.vasp import train

    calls = []
    monkeypatch.setattr(train, "train", lambda *args: calls.append(("train", args)))
    monkeypatch.setattr(
        train, "evaluate", lambda *args: calls.append(("evaluate", args))
    )
    main(["training-validation", "fit", "study", "full", "--epochs", "7"])
    assert calls == [
        ("train", ("study", "full", 7)),
        ("evaluate", ("study", "full")),
    ]


def test_worker_dispatch_preserves_integer_index_and_exceptions(monkeypatch):
    from mapho.workflows.vasp import atomic_reference

    def worker(root, index):
        assert root == "study" and index == 3
        raise ValueError("Reference validation failed")

    monkeypatch.setattr(atomic_reference, "worker", worker)
    with pytest.raises(ValueError, match="Reference validation failed"):
        main(["atomic-reference", "worker", "study", "3"])


def run_python(*args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *args], env=env, check=True, capture_output=True, text=True
    )


def test_core_inference_does_not_import_workflow_or_cli_layers():
    run_python(
        "-c",
        """
import sys
from mapho.vasp.checkpoints import load_joint_model, load_repaired_model
from mapho.vasp.train import load_model, predict
assert not any(name.startswith(('mapho.workflows', 'mapho.commands')) for name in sys.modules)
""",
    )


def test_top_level_help_does_not_import_electronic_or_training_backends():
    run_python(
        "-c",
        """
import sys
from mapho.commands.vasp import main
try:
    main(['--help'])
except SystemExit as result:
    assert result.code == 0
assert 'torch' not in sys.modules
assert not any(name.startswith(('mapho.vasp', 'mapho.workflows')) for name in sys.modules)
""",
    )


@pytest.mark.parametrize(
    "module",
    [
        "mapho.commands.vasp.silicon_dataset",
        "mapho.vasp.silicon_dataset",
        "mapho.vasp_silicon_dataset",
        "mlham.vasp_silicon_dataset",
    ],
)
def test_saved_worker_commands_remain_executable(module):
    result = run_python("-m", module, "--help")
    assert "convergence" in result.stdout and "worker" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_previous_checkpoint_loaders_reexport_core_functions():
    from mapho.vasp.checkpoints import load_joint_model, load_repaired_model

    for prefix in ("mapho.vasp.", "mapho.vasp_", "mlham.vasp_"):
        assert (
            importlib.import_module(prefix + "paw_joint").load_model is load_joint_model
        )
        assert (
            importlib.import_module(prefix + "paw_repair").load_model
            is load_repaired_model
        )
