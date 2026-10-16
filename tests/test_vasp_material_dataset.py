import numpy as np
import pytest
from mapho.experiments.vasp.material_dataset import structure, kpoints, prepare


def test_binary_bulk_and_slab_geometry_sampling():
    lif, _ = structure("lif")
    assert len(lif) == 16 and lif.composition.as_dict() == {"Li": 8.0, "F": 8.0}
    assert lif.volume == pytest.approx(2 * 4.03**3)
    slab, info = structure("mos2")
    assert len(slab) == 24 and slab.composition.as_dict() == {"S": 16.0, "Mo": 8.0}
    assert info["actual_outer_surface_gap_A"] == pytest.approx(20.0)
    assert len(set(np.round(slab.cart_coords[:, 2], 6))) == 6
    distorted, info = structure("mos2", 42)
    assert distorted.lattice.c == pytest.approx(slab.lattice.c)
    np.testing.assert_allclose(info["strain"][2], 0)
    for material in ("lif", "mos2"):
        base, _ = structure(material)
        meshes = [
            tuple(kpoints(base, d, material == "mos2")[1]["mesh"])
            for d in ((200, 300, 500) if material == "lif" else (500, 750, 1000))
        ]
        assert len(set(meshes)) == 3
        if material == "mos2":
            assert all(mesh[2] == 1 for mesh in meshes)
    taller, _ = structure("mos2", vacuum=30.0)
    assert kpoints(slab, 500, True)[1]["mesh"] == kpoints(taller, 500, True)[1]["mesh"]


@pytest.mark.parametrize("material", ["lif", "mos2"])
def test_inputs_and_split_gate(tmp_path, material):
    from pathlib import Path
    from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints

    for symbol, name, z in [
        ("Li", "Li_sv", 3),
        ("F", "F", 7),
        ("Mo", "Mo_pv", 12),
        ("S", "S", 6),
    ]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "POTCAR").write_text(
            f"TITEL = PAW_PBE {name} test\nLEXCH = PE\nZVAL = {z}\nENMAX = 400\nEnd of Dataset\n"
        )
    exe = tmp_path / "vasp"
    exe.write_text("synthetic identity, never executed")
    root = tmp_path / "run"
    m = prepare(root, material, tmp_path, exe, count=8)
    from mapho.source_snapshot import source_hashes

    assert m["source_sha256"] == source_hashes(root / "source_snapshot/mapho")
    assert "experiments/vasp/material_dataset.py" in m["source_sha256"]
    members = sum(m["splits"].values(), [])
    assert len(members) == len(set(members)) == 8
    assert not (root / "completion.json").exists()
    for row in m["samples"]:
        folder = Path(row["directory"])
        incar = Incar.from_file(folder / "INCAR")
        geom = Poscar.from_file(folder / "POSCAR").structure
        assert len(geom) == m["natoms"]
        assert incar["ISPIN"] == 1 and incar["NBANDS"] > m["nelect"] / 2
        if material == "mos2":
            assert Kpoints.from_file(folder / "KPOINTS").kpts[0][2] == 1
        assert (folder / "POTCAR").read_text().count("End of Dataset") == 2
