"""PBE0 sampling uses common k points, PAW gauge consistency, and a strict gate."""

import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mapho.experiments.vasp import pbe0_convergence as study
from mapho.vasp.paw import save_json, sha


def bands(mesh, shift=0.0):
    points = np.array(list(itertools.product(np.arange(mesh) / mesh, repeat=3)))
    energy = np.arange(12)[None, :] + np.cos(2 * np.pi * points).sum(1)[:, None]
    return dict(kpoints=points, eigenvalues_eV=energy + shift)


def test_frontier_compares_common_points_with_a_consistent_gauge():
    coarse, fine = bands(4), bands(8, shift=2.5)
    fine = {key: value[::-1] for key, value in fine.items()}
    np.testing.assert_allclose(
        study.common_frontier(coarse), study.common_frontier(fine)
    )
    fine["eigenvalues_eV"][:, 4] += 0.02
    assert np.max(
        np.abs(study.common_frontier(fine) - study.common_frontier(coarse))
    ) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="common k point"):
        study.common_frontier(bands(3))


def test_prepare_separates_full_hybrid_scf_from_replay(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "TITEL = PAW_PBE Si test\nLEXCH = PE\nZVAL = 4.0\nENMAX = 250\nEnd of Dataset\n"
    )
    executable = tmp_path / "vasp"
    executable.write_text("synthetic identity, never executed")
    root = tmp_path / "study"
    manifest = study.prepare(root, potcar, executable)
    assert len(manifest["cases"]) == 9
    for case in manifest["cases"]:
        folder = root / case["directory"] / "pbe0"
        m = study.read_json(folder / "manifest.json")
        settings = m["settings"]
        assert settings["LHFCALC"] and settings["LFOCKACE"]
        assert settings["HFSCREEN"] == settings["HFRCUT"] == 0
        assert settings["ICHARG"] == 0
        assert settings["ISYM"] == -1
        assert (folder / "MAPHO_FOCK_EXPORT").is_file()
        assert (folder / "MAPHO_EXPORT").is_file()
        assert not list(folder.glob("MLHAM*"))
        assert not (folder / "MAPHO_ACE_IMPORT").exists()
    assert study.collect(root)["selected_mesh"] is None
    assert not study.collect(root)["complete"]
    assert "experiments/vasp/pbe0_convergence.py" in manifest["source_sha256"]
    with pytest.raises(ValueError, match="even meshes"):
        study.prepare(tmp_path / "bad", potcar, executable, meshes=[3, 4, 6])


def write_case(folder, gauge):
    folder.mkdir()
    (folder / "pbe0").mkdir()
    xml = folder / "pbe0/vasprun.xml"
    xml.write_text(
        '<modeling><calculation><energy><i name="e_0_energy">-10</i></energy></calculation></modeling>'
    )
    path = folder / "operator_components.h5"
    q = np.array([[[0.2]], [[0.3]]])
    with h5py.File(path, "w") as f:
        f["cell_A"] = np.eye(3)
        f["positions_A"] = [[0, 0, 0], [0.25, 0.25, 0.25]]
        f["atomic_numbers"] = [14, 14]
        f["paw/types"] = [0, 0]
        f["paw/q"] = q
        f["paw/d"] = np.ones_like(q) + gauge * q
        f["potential_eV"] = np.arange(8).reshape(2, 2, 2) + gauge
    save_json(
        folder / "status.json",
        dict(
            complete=True,
            passed=True,
            output_sha256={
                "operator_components.h5": sha(path),
                "pbe0/vasprun.xml": sha(xml),
            },
        ),
    )


def test_comparison_removes_matching_potential_and_paw_gauge(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    write_case(a, 0.3)
    write_case(b, 0.0)
    monkeypatch.setattr(
        study,
        "read_bands",
        lambda folder: bands(4, 0.3) if folder.parent == a else bands(8),
    )
    result = study.compare_meshes(a, b)
    assert result["passed"]
    assert result["potential_rmse_eV"] < 1e-14
    assert result["onsite_rmse_eV"] < 1e-14
    (a / "pbe0/vasprun.xml").write_text("changed")
    with pytest.raises(ValueError, match="result changed"):
        study.compare_meshes(a, b)


@pytest.mark.parametrize("fine_passes", [True, False])
def test_gate_requires_finest_pair_on_every_geometry(
    tmp_path, monkeypatch, fine_passes
):
    cases = []
    for label in ("equilibrium", "compressed", "expanded_displaced"):
        for mesh in (4, 6, 8):
            directory = f"{label}/mesh_{mesh}"
            (tmp_path / directory).mkdir(parents=True)
            save_json(tmp_path / directory / "status.json", dict(complete=True))
            cases.append(dict(directory=directory, structure=label, mesh=mesh))
    save_json(
        tmp_path / "manifest.json",
        dict(cases=cases, meshes=[4, 6, 8], thresholds=study.THRESHOLDS),
    )

    def compare(a, b, thresholds):
        return {
            "passed": fine_passes
            if a.name == "mesh_6" and a.parent.name == "compressed"
            else True
        }

    monkeypatch.setattr(study, "compare_meshes", compare)
    result = study.collect(tmp_path)
    assert result["passed"] == fine_passes
    assert result["selected_mesh"] == (8 if fine_passes else None)
