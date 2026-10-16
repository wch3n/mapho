"""Conditional training must retain the hybrid reference contract and mesh gate."""

import json

import h5py
import numpy as np
import pytest

from mapho.experiments.vasp import pbe0_learning as study
from mapho.vasp.paw import COMPONENT_SCHEMA, SCHEMA, save_json, sha
from mapho.vasp.train import target_contract
from test_vasp_ace import hybrid_archive


def test_hybrid_targets_require_explicit_opt_in_and_paired_exchange(tmp_path):
    operator = hybrid_archive(tmp_path)
    baseline = tmp_path / "atomic.h5"
    with h5py.File(baseline, "w") as f:
        f.attrs["schema"] = SCHEMA
    target = tmp_path / "components.h5"
    manifest = {"hybrid_operators": {target.name: operator.name}}
    with h5py.File(target) as t, h5py.File(baseline) as b:
        with pytest.raises(ValueError, match="scalar PBE"):
            target_contract(t, b, target, tmp_path, manifest, "pbe")
        with pytest.raises(ValueError, match="paired reference"):
            target_contract(t, b, target, tmp_path, {}, "pbe0_components")
        result = target_contract(t, b, target, tmp_path, manifest, "pbe0_components")
        assert result["reference_exchange_operator"] == str(operator)
        assert result["reference_exchange_operator_sha256"] == sha(operator)
    with (tmp_path / "factors.bin").open("ab") as f:
        f.write(b"modified")
    with h5py.File(target) as t, h5py.File(baseline) as b:
        with pytest.raises(ValueError, match="exchange changed"):
            target_contract(t, b, target, tmp_path, manifest, "pbe0_components")


def test_exchange_cannot_be_paired_with_other_component_targets(tmp_path):
    operator = hybrid_archive(tmp_path)
    baseline = tmp_path / "atomic.h5"
    with h5py.File(baseline, "w") as f:
        f.attrs["schema"] = SCHEMA
    target = tmp_path / "components.h5"
    with h5py.File(target, "r+") as f:
        f["potential_eV"][0, 0, 0] += 0.1
    manifest = {"hybrid_operators": {target.name: operator.name}}
    with h5py.File(target) as t, h5py.File(baseline) as b:
        with pytest.raises(ValueError, match="different component"):
            target_contract(t, b, target, tmp_path, manifest, "pbe0_components")
    with h5py.File(baseline, "r+") as f:
        f.attrs["schema"] = COMPONENT_SCHEMA
    with h5py.File(target) as t, h5py.File(baseline) as b:
        with pytest.raises(ValueError, match="PBE atomic baseline"):
            target_contract(t, b, target, tmp_path, manifest, "pbe0_components")


def test_dataset_generation_requires_mesh_convergence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "collect", lambda root: dict(complete=True, passed=False)
    )
    with pytest.raises(ValueError, match="convergence must pass"):
        study.prepare_dataset(tmp_path / "dataset", tmp_path / "calibration")
    assert not (tmp_path / "dataset").exists()


def test_dataset_has_disjoint_geometry_splits_and_no_reference_inputs(
    tmp_path, monkeypatch
):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "TITEL = PAW_PBE Si test\nLEXCH = PE\nZVAL = 4.0\nENMAX = 250\nEnd of Dataset\n"
    )
    executable = tmp_path / "vasp"
    executable.write_text("synthetic identity, never executed")
    save_json(
        calibration / "manifest.json",
        dict(
            executable=str(executable),
            executable_sha256=sha(executable),
            potcar=str(potcar),
            potcar_sha256=sha(potcar),
        ),
    )
    gate = dict(complete=True, passed=True, selected_mesh=8)
    save_json(calibration / "convergence.json", gate)
    monkeypatch.setattr(study, "collect", lambda root: gate)
    root = tmp_path / "dataset"
    manifest = study.prepare_dataset(root, calibration)
    partitions = [set(manifest["splits"][s]) for s in ("train", "validation", "test")]
    assert [len(p) for p in partitions] == [16, 4, 4]
    assert len(set.union(*partitions)) == 24
    assert manifest["correction_activation"] == "gelu"
    cells = []
    for index, case in enumerate(manifest["cases"]):
        folder = root / case["directory"]
        pbe = json.loads((folder / "pbe/manifest.json").read_text())["settings"]
        hybrid = json.loads((folder / "pbe0/manifest.json").read_text())["settings"]
        assert pbe["ALGO"] == "Normal"
        assert hybrid["ICHARG"] == 0 and hybrid["HFSCREEN"] == 0
        assert (folder / "POSCAR").read_bytes() == (folder / "pbe0/POSCAR").read_bytes()
        assert not (folder / "pbe0/WAVECAR").exists()
        cells.append(study.structure(index).cell.array)
    assert len({c.tobytes() for c in cells}) == 24
    np.testing.assert_array_equal(cells[7], study.structure(7).cell.array)
    assert not study.audit(root)["passed"]
    with pytest.raises(ValueError, match="must pass before training"):
        study.prepare_training(tmp_path / "training", root)
