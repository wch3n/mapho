"""Backend relocation preserves saved imports and isolated worker execution."""

import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from mapho._compat import BACKEND_ALIASES
from mapho.source_snapshot import snapshot_package, source_hashes


@pytest.mark.parametrize("old,new", sorted(BACKEND_ALIASES.items()))
def test_flat_backend_aliases_share_the_canonical_module(old, new):
    if new.endswith("_patch"):
        pytest.importorskip(new)  # Private patchers are absent from releases.
    canonical = importlib.import_module(new)
    assert importlib.import_module(old) is canonical
    assert importlib.import_module(old.replace("mapho.", "mlham.", 1)) is canonical
    assert importlib.import_module(new.replace("mapho.", "mlham.", 1)) is canonical
    assert canonical.__name__ == canonical.__spec__.name == new


@pytest.mark.parametrize(
    "module,name,new",
    [
        ("mapho.qe", "QEConfig", "mapho.qe.backend"),
        ("mapho.vasp_model", "PAWModel", "mapho.vasp.model"),
        (
            "mapho.vasp_paw_conditioning",
            "ConditionedPAWModel",
            "mapho.vasp.paw_conditioning",
        ),
        ("mlham.qe", "QEConfig", "mapho.qe.backend"),
        ("mlham.vasp_model", "PAWModel", "mapho.vasp.model"),
    ],
)
def test_historical_pickle_class_paths(module, name, new):
    # Protocol-0 GLOBAL payloads reproduce the original flat class locations.
    payload = f"c{module}\n{name}\n.".encode()
    assert pickle.loads(payload) is getattr(importlib.import_module(new), name)


@pytest.mark.parametrize(
    "module",
    [
        "mapho.qe",
        "mapho.qe.backend",
        "mapho.qe.exchange",
        "mapho.qe_exchange",
        "mapho.vasp.paw",
        "mapho.vasp_paw",
        "mlham.vasp_paw",
    ],
)
def test_canonical_and_historical_commands(module, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "usage:" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_snapshot_runs_nested_backends_without_source_tree_fallback(tmp_path):
    package = tmp_path / "source_snapshot/mapho"
    hashes = snapshot_package(package)
    assert hashes == source_hashes(package)
    assert {
        "_compat.py",
        "source_snapshot.py",
        "field_model.py",
        "qe/backend.py",
        "qe/files.py",
        "vasp/model.py",
        "vasp/paw.py",
        "vasp/silicon_dataset.py",
    } <= hashes.keys()
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
