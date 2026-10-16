import json

import h5py
import numpy as np
import pytest
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

from mapho.workflows.vasp.silicon_dataset import (
    silicon_structure,
    density_kpoints,
    prepare,
    compare_operators,
    convergence,
)


def test_supercells_are_sixteen_atoms_with_independent_internal_displacements():
    pristine, _ = silicon_structure()
    distorted, info = silicon_structure(42)
    assert len(pristine) == len(distorted) == 16
    assert pristine.volume == pytest.approx(2 * 5.43**3)
    assert np.linalg.norm(np.array(info["displacements_A"]), axis=1).min() > 0
    np.testing.assert_allclose(np.mean(info["displacements_A"], axis=0), 0, atol=1e-16)
    assert info["minimum_distance_A"] > 2.1
    assert density_kpoints(pristine, 500)[1]["mesh"] == [7, 7, 7]
    assert density_kpoints(distorted, 500)[1]["mesh"] == [7, 7, 7]


def test_written_inputs_share_density_and_splits_exclude_calibration(tmp_path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "TITEL = PAW_PBE Si test\nLEXCH = PE\nZVAL = 4.0\nENMAX = 250\nEnd of Dataset\n"
    )
    executable = tmp_path / "vasp"
    executable.write_text("synthetic identity, never executed")
    root = tmp_path / "study"
    m = prepare(root, potcar, executable, count=8)
    from mapho.source_snapshot import source_hashes

    assert m["source_sha256"] == source_hashes(root / "source_snapshot/mapho")
    assert "workflows/vasp/silicon_dataset.py" in m["source_sha256"]
    assert "_compat.py" in m["source_sha256"]
    members = sum(m["splits"].values(), [])
    assert len(members) == len(set(members)) == 8
    assert all(path.startswith("samples/") for path in members)
    for sample in m["samples"]:
        from pathlib import Path

        folder = Path(sample["directory"])
        assert len(Poscar.from_file(folder / "POSCAR").structure) == 16
        assert Kpoints.from_file(folder / "KPOINTS").kpts[0] == (7, 7, 7)
        assert Incar.from_file(folder / "INCAR")["ISPIN"] == 1
        assert sample["kpoints"]["requested_kppvol"] == 500
    assert not (root / "completion.json").exists()
    assert not convergence(root)["passed"]


def make_operator(folder, potential=0.0, d=0.0, eigen_shift=0.0, position=0.0, k=0.0):
    folder.mkdir()
    (folder / "report.json").write_text(
        json.dumps(dict(complete=True, energy_eV=-10.0, nocc=4, sampled_gap_eV=1.0))
    )
    with h5py.File(folder / "operator.h5", "w") as f:
        f["cell_A"] = np.eye(3)
        f["positions_A"] = [[position, 0, 0]]
        f["atomic_numbers"] = [14]
        f["paw/types"] = [0]
        f["paw/q"] = [[[0.2]]]
        f["paw/d"] = [[[d]]]
        f["paw"].attrs["channels_json"] = "[[0]]"
        f["potential_eV"] = np.full((2, 2, 2), potential)
        f["bands/eigenvalues_eV"] = [np.arange(8) + eigen_shift]
        f["bands/kpoints"] = [[k, 0, 0]]


@pytest.mark.parametrize(
    "change", [dict(potential=0.006), dict(d=0.006), dict(eigen_shift=0.002)]
)
def test_convergence_cannot_pass_from_energy_alone(tmp_path, change):
    make_operator(tmp_path / "a")
    make_operator(tmp_path / "b", **change)
    thresholds = dict(
        energy_eV_per_atom=0.001,
        common_k_frontier_max_eV=0.001,
        potential_rmse_eV=0.005,
        onsite_rmse_eV=0.005,
    )
    result = compare_operators(tmp_path / "a", tmp_path / "b", thresholds)
    assert result["energy_eV_per_atom"] == 0
    assert not result["passed"]


@pytest.mark.parametrize("change", [dict(position=0.01), dict(k=0.25)])
def test_convergence_rejects_different_geometry_or_missing_common_gamma(
    tmp_path, change
):
    make_operator(tmp_path / "a")
    make_operator(tmp_path / "b", **change)
    with pytest.raises(ValueError):
        compare_operators(tmp_path / "a", tmp_path / "b", {})
