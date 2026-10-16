"""Historical imports/data remain readable without duplicate implementations."""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mapho.fields import load_operator_reference
from mapho.qe_files import save_directory
from test_fields import example


def run_python(*args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *args], env=env, text=True, capture_output=True, check=True
    )


def test_lazy_alias_preserves_class_module_and_pickle_identity():
    run_python(
        "-c",
        """
import importlib, pickle, sys
import mlham
assert 'torch' not in sys.modules
assert 'mapho.qe' not in sys.modules
canonical = importlib.import_module('mapho.qe')
legacy = importlib.import_module('mlham.qe')
assert legacy is canonical
assert canonical.__name__ == canonical.__spec__.name == 'mapho.qe'
assert legacy.QEConfig is canonical.QEConfig
payload = pickle.dumps(canonical.QEConfig, protocol=0).replace(b'mapho.qe', b'mlham.qe')
assert pickle.loads(payload) is canonical.QEConfig
from mlham import qe
assert qe is canonical
assert importlib.import_module('mlham.schema') is importlib.import_module('mapho.schema')
""",
    )


@pytest.mark.parametrize("module", ["mlham", "mlham.qe"])
def test_legacy_module_commands(module):
    result = run_python("-m", module, "--help")
    assert "usage:" in result.stdout and "--help" in result.stdout


@pytest.mark.parametrize("prefix", ["mapho", "mlham"])
def test_old_and_new_field_files_load_without_rewriting(tmp_path, prefix):
    sample = example()
    path = tmp_path / "fields.npz"
    sample.save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    metadata = json.loads(str(arrays["metadata"]))
    assert metadata["schema"] == "mapho.scalar_fields.v1"
    metadata["schema"] = prefix + ".scalar_fields.v1"
    arrays["metadata"] = json.dumps(metadata)
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_operator_reference(path)
    np.testing.assert_array_equal(loaded.hxc, sample.hxc)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    loaded.save(tmp_path / "new.npz")
    with np.load(tmp_path / "new.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["schema"] == "mapho.scalar_fields.v1"


@pytest.mark.parametrize(
    "schema", ["mlham.scalar_fields.v999", "other.scalar_fields.v1"]
)
def test_namespace_compatibility_does_not_accept_unknown_formats(schema):
    sample = example()
    sample.metadata["schema"] = schema
    with pytest.raises(ValueError, match="schema"):
        sample.validate()


@pytest.mark.parametrize("prefix", ["mapho", "mlham", "custom_si"])
def test_qe_restart_uses_explicit_input_prefix(tmp_path, prefix):
    (tmp_path / "qe.in").write_text(f"&CONTROL\n prefix = '{prefix}'\n/\n")
    # Stale saves may coexist; do not choose a directory by glob/order.
    for name in ("mapho", "mlham"):
        (tmp_path / "scratch" / f"{name}.save").mkdir(parents=True)
    assert save_directory(tmp_path) == tmp_path / "scratch" / f"{prefix}.save"


@pytest.mark.parametrize("prefix", [None, "../escape", "/absolute", ".."])
def test_missing_or_invalid_qe_prefix_fails_closed(tmp_path, prefix):
    text = "" if prefix is None else f" prefix = '{prefix}'\n"
    (tmp_path / "qe.in").write_text("&CONTROL\n" + text + "/\n")
    with pytest.raises(ValueError, match="prefix"):
        save_directory(tmp_path)
