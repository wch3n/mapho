"""Mesh geometry and convergence decisions, including false-positive guards."""

import copy
import json

from ase import Atoms
import numpy as np
import pytest

from mapho.kpoints import mesh_from_spacing
from mapho.qe.kpoint_convergence import compare_spectra, summarize
from mapho.qe.supercell_reference import structure


def test_anisotropic_cell_gets_directional_reciprocal_sampling():
    for repeats in (4, 6, 8):
        atoms, _ = structure(repeats, "pristine")
        assert mesh_from_spacing(atoms, 0.8)["mesh"] == [1, 2, 2]
        assert mesh_from_spacing(atoms, 0.4)["mesh"] == [1, 4, 4]
        fine = mesh_from_spacing(atoms, 0.4)
        long_spacing = fine["actual_spacing_inv_angstrom"][0] / 2 * (1 + 1e-12)
        assert mesh_from_spacing(atoms, [long_spacing, 0.4, 0.4])["mesh"] == [2, 4, 4]


def test_skew_cell_and_rotation_use_reciprocal_metric():
    atoms = Atoms(
        "Si",
        scaled_positions=[[0, 0, 0]],
        cell=[[8, 0, 0], [7, 2, 0], [0, 0, 20]],
        pbc=True,
    )
    proposal = mesh_from_spacing(atoms, 0.4)
    expected = np.ceil(
        np.linalg.norm(2 * np.pi * np.linalg.inv(atoms.cell).T, axis=1) / 0.4
    )
    np.testing.assert_array_equal(proposal["mesh"], expected)
    assert np.all(np.array(proposal["actual_spacing_inv_angstrom"]) <= 0.4)
    atoms.rotate(31, [1, 2, 3], rotate_cell=True)
    assert mesh_from_spacing(atoms, 0.4)["mesh"] == proposal["mesh"]


@pytest.mark.parametrize("spacing", [0, -1, float("nan"), [0.4, 0.4, 0]])
def test_invalid_spacing_rejected(spacing):
    atoms, _ = structure(4, "pristine")
    with pytest.raises(ValueError):
        mesh_from_spacing(atoms, spacing)


def report(eigen=((0.0, 2.0),), k=((0.0, 0.0, 0.0),), gap=2.0, energy=-10.0):
    return dict(
        eigenvalues_eV=eigen,
        kpoints_cartesian_2pi_over_alat=k,
        sampled_gap_eV=gap,
        total_energy_hartree=energy,
    )


def test_common_k_matching_ignores_order_and_rejects_uniform_energy_drift():
    a = report()
    b = report(eigen=[[4.0, 7.0], [0.0002, 2.0002]], k=[[0.5, 0, 0], [0, 0, 0]])
    result = compare_spectra(a, b, 1, 32)
    assert result["passed"] and result["common_k_count"] == 1
    b["eigenvalues_eV"][1] = [0.002, 2.002]
    assert not compare_spectra(a, b, 1, 32)["passed"]
    # Constant gap alone does not hide an unconverged potential gauge/level.
    assert compare_spectra(a, b, 1, 32)["sampled_gap_change_eV"] == 0


def test_missing_gamma_or_changed_sampled_extremum_cannot_pass():
    with pytest.raises(ValueError):
        compare_spectra(report(), report(k=[[0.5, 0, 0]]), 1, 32)
    assert not compare_spectra(report(), report(gap=2.01), 1, 32)["passed"]


def test_summary_requires_both_functionals_and_every_refinement(tmp_path):
    cases = [
        dict(name=str(i), kmesh=mesh, nocc=1, natoms=32)
        for i, mesh in enumerate(([1, 2, 2], [1, 3, 3], [1, 4, 4], [2, 4, 4]))
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(series=[dict(name="test", cases=cases)], tolerance_eV=0.001))
    )
    assert not summarize(tmp_path)["all_spectral_checks_passed"]
    for case in cases:
        for functional in ("PBE", "PBE0"):
            folder = tmp_path / "cases" / case["name"] / functional
            folder.mkdir(parents=True)
            (folder / "report.json").write_text(json.dumps(report()))
    summary = summarize(tmp_path)
    assert summary["all_spectral_checks_passed"]
    assert summary["series"][0]["selected_mesh"] == [2, 4, 4]
    assert not summary["training_ready"]  # Field convergence still needs validation.
    (tmp_path / "cases/0/PBE0/report.json").write_text(json.dumps(report(gap=2.1)))
    assert not summarize(tmp_path)["all_spectral_checks_passed"]


def test_reuse_allows_only_coordinate_serialization_roundoff(tmp_path):
    from mapho.qe.kpoint_convergence import equivalent_inputs

    text = """&CONTROL
 prefix = 'mlham'
 outdir = '/old'
 max_seconds = 6000
/
&SYSTEM
 nat = 1
 ntyp = 1
 ecutwfc = 100
/
ATOMIC_SPECIES
Li 6.94 Li.upf
K_POINTS automatic
1 3 3 0 0 0
CELL_PARAMETERS angstrom
10 0 0
0 4 0
0 0 4
ATOMIC_POSITIONS angstrom
Li 0.0144646173 0 0
"""
    a, b = tmp_path / "a.in", tmp_path / "b.in"
    a.write_text(text)
    b.write_text(
        text.replace("/old", "/new")
        .replace("6000", "72000")
        .replace("mlham", "mapho")
        .replace("0.0144646173", "0.0144646200")
    )
    assert equivalent_inputs(a, b) == pytest.approx(2.7e-9)
    for changed in (
        text.replace("0.0144646173", "0.0145"),
        text.replace("1 3 3 0 0 0", "1 4 4 0 0 0"),
        text.replace("ecutwfc = 100", "ecutwfc = 90"),
    ):
        b.write_text(changed)
        with pytest.raises(RuntimeError):
            equivalent_inputs(a, b)


def test_launcher_timeout_is_recorded_as_failed_not_running(tmp_path, monkeypatch):
    import subprocess
    from mapho.qe.kpoint_convergence import _launch
    from mapho.qe.supercell_reference import digest

    folder = tmp_path / "cases/test/PBE"
    folder.mkdir(parents=True)
    original = folder / "qe.in"
    original.write_text("&CONTROL\n max_seconds = 900\n/\n")
    case = dict(name="test", inputs={"PBE": {"sha256": digest(original)}})
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="no clean checkpoint"):
        _launch(
            tmp_path,
            {"executable": "/fake/pw.x"},
            case,
            "PBE",
            128,
            120,
            grace_seconds=300,
        )
    state = json.loads((folder / "state.json").read_text())
    assert calls == [420]
    assert state["state"] == state["attempts"][-1]["state"] == "failed"
    with pytest.raises(RuntimeError, match="requires inspection"):
        _launch(tmp_path, {"executable": "/fake/pw.x"}, case, "PBE", 128, 120)
