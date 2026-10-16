"""Isolated worker snapshots contain the canonical package and experiment sources."""

from importlib.metadata import distribution
import json
import os
import subprocess
import sys

import pytest

from mapho.source_snapshot import snapshot_package, source_hashes


def test_snapshot_runs_nested_backends_without_source_tree_fallback(tmp_path):
    package = tmp_path / "source_snapshot/mapho"
    hashes = snapshot_package(package)
    assert hashes == source_hashes(package)
    assert {
        "source_snapshot.py",
        "field_model.py",
        "qe/backend.py",
        "qe/files.py",
        "vasp/model.py",
        "vasp/paw.py",
        "vasp/checkpoints.py",
        "commands/vasp/predict.py",
        "experiments/vasp/silicon_dataset.py",
    } <= hashes.keys()
    assert "_compat.py" not in hashes
    assert not any(name.startswith("workflows/") for name in hashes)
    assert not list(package.rglob("*_patch.py"))
    assert not list(package.rglob("*.inc"))
    with pytest.raises(FileExistsError):
        snapshot_package(package)
    modules = [
        name[:-3].replace("/", ".")
        for name in hashes
        if not name.endswith(("__init__.py", "__main__.py"))
    ]
    script = """
import importlib, json, pathlib, sys
import mapho
from mapho.source_snapshot import source_hashes, source_path
package = pathlib.Path(sys.argv[1])
assert pathlib.Path(mapho.__file__).resolve().parent == package
for module in json.loads(sys.argv[2]):
    importlib.import_module('mapho.' + module)
for name, module in list(sys.modules.items()):
    if name.startswith('mapho.') and getattr(module, '__file__', None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(package), name
assert source_hashes() == json.loads(sys.argv[3])
assert source_path('vasp/paw.py') == package/'vasp/paw.py'
"""
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(package),
            json.dumps(modules),
            json.dumps(hashes),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    for group, command in (
        ("data", "silicon"),
        ("train", "paw-joint"),
        ("predict", "binary-bands"),
        ("validate", "ace"),
    ):
        result = subprocess.run(
            [sys.executable, "-m", "mapho.commands.vasp." + group, command, "--help"],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_installed_console_entry_points_resolve_without_import_aliases():
    scripts = [
        entry
        for entry in distribution("mapho").entry_points
        if entry.group == "console_scripts"
    ]
    assert {entry.name for entry in scripts} == {
        "mapho",
        "mapho-qe",
        "mapho-fields",
        "mapho-vasp",
    }
    for entry in scripts:
        assert callable(entry.load()), entry.name
