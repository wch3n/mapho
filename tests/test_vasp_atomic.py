"""Check atomic export completion and provenance without licensed native inputs."""

import json
from types import SimpleNamespace

from ase.build import bulk
import h5py
import numpy as np
import pytest

from mapho.vasp import atomic, paw


@pytest.fixture
def reference(tmp_path):
    potcar = tmp_path / "POTCAR.synthetic"
    potcar.write_text(
        "TITEL = PAW_PBE Si test\nLEXCH = PE\nZVAL = 4.0\nENMAX = 250\nEnd of Dataset\n"
    )
    folder = tmp_path / "atomic"
    atomic.prepare_atomic_reference(
        folder, bulk("Si", "diamond", a=5.43), {"Si": potcar}, grid_shape=(2, 3, 4)
    )
    executable = tmp_path / "synthetic_executable"
    executable.write_text("Never executed by these tests.\n")
    return folder, executable


def native_outputs(folder, *, completion=True, scstep=False):
    paw.write_pot(
        folder / "POT",
        np.arange(24).reshape(2, 3, 4) * 0.1,
        (folder / "POSCAR").read_text(),
        "augmentation occupancies   1  1\n0\naugmentation occupancies   2  1\n0\n",
    )
    paw.write_paw(
        folder / "MAPHO_PAW.dat",
        dict(
            d=np.ones((2, 1, 1)),
            q=np.zeros((2, 1, 1)),
            types=[0, 0],
            channels=[[0]],
        ),
    )
    (folder / "vasprun.xml").write_text(
        "<modeling>"
        + ("<calculation><scstep/></calculation>" if scstep else "")
        + "</modeling>"
    )
    (folder / "OUTCAR").write_text("General timing and accounting\n")
    if completion:
        paw.save_json(folder / atomic.COMPLETION, atomic.COMPLETION_RECORD)


def mock_native(monkeypatch, *, completion=True, scstep=False):
    calls = []

    def execute(command, *, cwd, stdout, **kwargs):
        calls.append(command)
        native_outputs(cwd, completion=completion, scstep=scstep)
        stdout.write("MAPHO: full PAW D and Q exported\n" + atomic.MESSAGE + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(paw.subprocess, "run", execute)
    return calls


def test_band_free_operator_exports_and_replays(reference, monkeypatch, tmp_path):
    folder, executable = reference
    mock_native(monkeypatch)
    result = paw.run(folder, executable)
    assert result["status"] == "complete"
    assert result["diagonalized"] is False
    assert result["electronic_steps"] == 0
    assert not (folder / "EIGENVAL").exists()
    output = folder / "operator.h5"
    paw.export_operator(folder, output)
    with h5py.File(output) as archive:
        assert archive.attrs["schema"] == paw.SCHEMA
        assert "bands" not in archive
        np.testing.assert_allclose(
            archive["potential_eV"][:], np.arange(24).reshape(2, 3, 4) * 0.1
        )
    replay = paw.prepare_replay(output, tmp_path / "replay")
    assert replay["settings"]["ICHARG"] == 14
    assert not (tmp_path / "replay" / atomic.MARKER).exists()


@pytest.mark.parametrize("problem", ["missing_completion", "electronic_steps"])
def test_completion_requires_native_evidence(reference, monkeypatch, problem):
    folder, executable = reference
    mock_native(
        monkeypatch,
        completion=problem != "missing_completion",
        scstep=problem == "electronic_steps",
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        paw.run(folder, executable)
    assert json.loads((folder / "run.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="completed native reference"):
        paw.export_operator(folder, folder / "operator.h5")


@pytest.mark.parametrize(
    "name", ["CHGCAR", "WAVECAR", "POT", atomic.COMPLETION, "MAPHO_IMPORT"]
)
def test_restart_and_stale_outputs_rejected_before_launch(reference, monkeypatch, name):
    folder, executable = reference
    calls = mock_native(monkeypatch)
    (folder / name).touch()
    with pytest.raises(ValueError, match="start without"):
        paw.run(folder, executable)
    assert calls == []
    assert not (folder / "run.json").exists()


@pytest.mark.parametrize(
    "setting,value",
    [("ISPIN", 2), ("ICHARG", 14), ("LHFCALC", True), ("METAGGA", "SCAN")],
)
def test_unsupported_settings_rejected(reference, setting, value):
    folder, _ = reference
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["settings"][setting] = value
    with pytest.raises(ValueError, match="scalar PBE|Unsupported settings"):
        atomic.validate_request(folder, manifest)


def test_post_run_output_tampering_rejected(reference, monkeypatch):
    folder, executable = reference
    mock_native(monkeypatch)
    paw.run(folder, executable)
    with (folder / "POT").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="outputs changed"):
        paw.export_operator(folder, folder / "operator.h5")


def test_export_marker_cannot_bypass_regular_run_checks(reference, monkeypatch):
    folder, executable = reference
    calls = mock_native(monkeypatch)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest.pop("atomic_export_only")
    paw.save_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="export-only manifest"):
        paw.run(folder, executable)
    assert calls == []
