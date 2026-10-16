"""Grouped commands preserve dispatch and keep inference independent of studies."""

import importlib
import json
import os
import subprocess
import sys

import pytest

from mapho.commands.vasp import GROUPS, main

COMMANDS = [
    (group, command)
    for group in GROUPS
    for command in importlib.import_module("mapho.commands.vasp." + group).COMMANDS
]


@pytest.mark.parametrize("group,command", COMMANDS)
def test_each_command_has_help_without_running_calculations(group, command, capsys):
    with pytest.raises(SystemExit) as result:
        main([group, command, "--help"])
    assert result.value.code == 0
    assert f"usage: mapho-vasp {group} {command}" in capsys.readouterr().out


def test_paw_dispatch_preserves_positional_order_and_shift(monkeypatch, capsys):
    from mapho.vasp import paw

    calls = []

    def compare(*args):
        calls.append(args)
        return {"passed": True}

    monkeypatch.setattr(paw, "compare", compare)
    main(["predict", "paw", "compare", "reference", "prediction", "--shift-eV", "0.25"])
    assert calls == [("reference", "prediction", 0.25)]
    assert json.loads(capsys.readouterr().out) == {"passed": True}


def test_silicon_dispatch_uses_experiment_namespace(monkeypatch, capsys):
    from mapho.experiments.vasp import silicon_dataset

    monkeypatch.setattr(silicon_dataset, "convergence", lambda root: {"root": root})
    main(["data", "silicon", "convergence", "study"])
    assert json.loads(capsys.readouterr().out) == {"root": "study"}


def test_training_validation_calls_reusable_training_api(monkeypatch):
    from mapho.vasp import train

    calls = []
    monkeypatch.setattr(train, "train", lambda *args: calls.append(("train", args)))
    monkeypatch.setattr(
        train, "evaluate", lambda *args: calls.append(("evaluate", args))
    )
    main(["validate", "training", "fit", "study", "full", "--epochs", "7"])
    assert calls == [("train", ("study", "full", 7)), ("evaluate", ("study", "full"))]


def test_worker_dispatch_preserves_integer_index_and_exceptions(monkeypatch):
    from mapho.experiments.vasp import atomic_reference

    def worker(root, index):
        assert root == "study" and index == 3
        raise ValueError("Reference validation failed")

    monkeypatch.setattr(atomic_reference, "worker", worker)
    with pytest.raises(ValueError, match="Reference validation failed"):
        main(["data", "atomic-reference", "worker", "study", "3"])


def run_python(*args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *args], env=env, check=True, capture_output=True, text=True
    )


def test_core_inference_does_not_import_experiment_or_cli_layers():
    run_python(
        "-c",
        """
import sys
from mapho.vasp.checkpoints import load_joint_model, load_repaired_model
from mapho.vasp.train import load_model, predict
assert not any(name.startswith(('mapho.experiments', 'mapho.commands')) for name in sys.modules)
""",
    )


@pytest.mark.parametrize("args", [[], *[[group] for group in GROUPS]])
def test_group_help_does_not_import_electronic_or_training_backends(args):
    run_python(
        "-c",
        """
import json, sys
from mapho.commands.vasp import main
try:
    main(json.loads(sys.argv[1]) + ['--help'])
except SystemExit as result:
    assert result.code == 0
assert 'torch' not in sys.modules
assert not any(name.startswith(('mapho.vasp', 'mapho.experiments')) for name in sys.modules)
""",
        json.dumps(args),
    )


@pytest.mark.parametrize("args", [[], ["unknown"], ["data", "unknown"], ["train"]])
def test_missing_or_invalid_command_fails_before_backend_import(args):
    run_python(
        "-c",
        """
import json, sys
from mapho.commands.vasp import main
try:
    main(json.loads(sys.argv[1]))
except SystemExit as result:
    assert result.code == 2
else:
    raise AssertionError('Invalid command was accepted')
assert 'torch' not in sys.modules
""",
        json.dumps(args),
    )


def test_no_legacy_import_hook_is_installed():
    run_python(
        "-c",
        """
import importlib.util, sys
before = list(sys.meta_path)
import mapho
assert sys.meta_path == before
for name in ('mlham', 'mapho._compat', 'mapho.vasp_paw', 'mapho.workflows'):
    assert importlib.util.find_spec(name) is None, name
""",
    )
