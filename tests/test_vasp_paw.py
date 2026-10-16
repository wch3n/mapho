from io import StringIO
import json

from ase.build import bulk
from ase.io import write
import h5py
import numpy as np
import pytest

from mapho.vasp.paw import (
    SCHEMA,
    check_pot_geometry,
    prepare_replay,
    read_bands,
    read_paw,
    read_pot,
    write_paw,
    write_pot,
)
from mapho.file_utils import sha


def test_potential_xyz_order_units_and_native_suffix(tmp_path):
    # Analytic ramp detects axis transpositions and accidental CHGCAR volume scaling.
    atoms = bulk("Si", "diamond", a=5.43)
    out = StringIO()
    write(out, atoms, format="vasp", direct=True, vasp5=True)
    header = out.getvalue()
    suffix = "augmentation occupancies   1  1\n0.125\naugmentation occupancies   2  1\n0.25\n"
    x, y, z = np.indices((3, 4, 5))
    potential = 0.1 * x + y + 10 * z
    p = tmp_path / "POT"
    write_pot(p, potential, header, suffix)
    structure, field, h, s = read_pot(p)
    np.testing.assert_allclose(field, potential, atol=1e-14)
    np.testing.assert_allclose(structure.positions, atoms.positions)
    assert s == suffix
    # Inspect the on-disk stream independently of our reader.
    lines = p.read_text().splitlines()
    start = lines.index("3 4 5") + 1
    assert list(map(float, lines[start].split())) == pytest.approx(
        [0.0, 0.1, 0.2, 1.0, 1.1]
    )
    assert len(lines[start]) == 5 * 18
    assert [
        float(lines[start][i * 18 + 1 : (i + 1) * 18]) for i in range(5)
    ] == pytest.approx([0.0, 0.1, 0.2, 1.0, 1.1])
    assert h.strip() == header.strip()


def test_paw_channels_padding_and_gauge_shift(tmp_path):
    channels = [[0, 1, 0], [0, 0]]
    rng = np.random.default_rng(4)
    d = np.zeros((2, 5, 5))
    q = np.zeros_like(d)
    for a, n in enumerate([5, 2]):
        r = rng.normal(size=(n, n))
        d[a, :n, :n] = (r + r.T) / 2
        r = rng.normal(size=(n, n))
        q[a, :n, :n] = (r + r.T) / 2
    paw = dict(d=d, q=q, types=np.array([0, 1]), channels=channels)
    p = tmp_path / "MAPHO_PAW.dat"
    write_paw(p, paw)
    assert p.read_text().splitlines()[0] == "MAPHO_PAW_V1"
    other = read_paw(p)
    assert other["channels"] == channels
    np.testing.assert_allclose(other["d"], d, atol=1e-16)
    np.testing.assert_allclose(other["q"], q, atol=1e-16)
    # Test delta H = c S using an independent finite plane-wave/projector model.
    projectors = rng.normal(size=(9, 5)) * 0.1
    s = np.eye(9) + projectors @ q[0] @ projectors.T
    shift = 0.25
    delta_h = shift * np.eye(9) + projectors @ (shift * other["q"][0]) @ projectors.T
    np.testing.assert_allclose(delta_h, shift * s, atol=1e-15)


@pytest.mark.parametrize("bad", ["asymmetric", "nan", "complex"])
def test_invalid_onsite_matrix_rejected(tmp_path, bad):
    d = np.zeros((1, 2, 2), dtype=complex if bad == "complex" else float)
    if bad == "asymmetric":
        d[0, 0, 1] = 0.1
    if bad == "nan":
        d[0, 0, 0] = np.nan
    if bad == "complex":
        d[0, 0, 0] = 1j
    with pytest.raises(ValueError):
        write_paw(
            tmp_path / "bad",
            dict(d=d, q=np.zeros((1, 2, 2)), types=[0], channels=[[0, 0]]),
        )


def test_patch_rejects_ambiguous_version_anchor():
    replace_once = pytest.importorskip("mapho.vasp.vasp_patch").replace_once
    with pytest.raises(ValueError):
        replace_once("abc abc", "abc", "x")
    with pytest.raises(ValueError):
        replace_once("def", "abc", "x")


@pytest.mark.parametrize("schema", [SCHEMA, "mlham.vasp_paw_operator.v1"])
def test_replay_api_preserves_paw_gauge_and_checks_dataset(tmp_path, schema):
    atoms = bulk("Si", "diamond", a=5.43)
    ref = tmp_path / "ref"
    ref.mkdir()
    write(ref / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    # Synthetic sentinel, never supplied to a real VASP executable.
    (ref / "POTCAR").write_text("unit-test PAW dataset identity")
    (ref / "KPOINTS").write_text("unit-test k mesh")
    header = (ref / "POSCAR").read_text()
    suffix = "augmentation occupancies   1  1\n0\naugmentation occupancies   2  1\n0\n"
    metadata = dict(
        reference=str(ref),
        reference_manifest=dict(
            settings={},
            inputs={n: sha(ref / n) for n in ("POSCAR", "POTCAR", "KPOINTS")},
        ),
    )
    operator = tmp_path / "operator.h5"
    q = np.array([[[0.2]], [[0.1]]])
    d = np.array([[[1.0]], [[2.0]]])
    with h5py.File(operator, "w") as f:
        f.attrs["schema"] = schema
        f.attrs["metadata_json"] = json.dumps(metadata)
        f["potential_eV"] = np.ones((2, 3, 4)) * 3
        f["paw/d"] = d
        f["paw/q"] = q
        f["paw/types"] = [0, 0]
        f["paw"].attrs["channels_json"] = "[[0]]"
        f["native_pot_header"] = header
        f["native_pot_suffix"] = suffix
    prepare_replay(operator, tmp_path / "replay", shift_eV=0.25)
    assert (tmp_path / "replay/MAPHO_IMPORT").is_file()
    assert not list((tmp_path / "replay").glob("MLHAM*"))
    _, v, _, _ = read_pot(tmp_path / "replay/POT")
    paw = read_paw(tmp_path / "replay/MAPHO_PAW.dat")
    # Independent generalized eigenproblem, built from serialized API output.
    p = np.array([[0.3, 0.1], [0.2, 0.4], [0.1, 0.2]])
    overlap = np.eye(3) + p @ np.diag(q[:, 0, 0]) @ p.T
    original = np.diag([3.0, 4.0, 5.0]) + p @ np.diag(d[:, 0, 0]) @ p.T
    replay = (
        np.diag([v[0, 0, 0], v[0, 0, 0] + 1, v[0, 0, 0] + 2])
        + p @ np.diag(paw["d"][:, 0, 0]) @ p.T
    )
    inv_l = np.linalg.inv(np.linalg.cholesky(overlap))
    a = np.linalg.eigvalsh(inv_l @ original @ inv_l.T)
    b = np.linalg.eigvalsh(inv_l @ replay @ inv_l.T)
    np.testing.assert_allclose(b - a, 0.25, atol=1e-14)
    assert "ICHARG = 14" in (tmp_path / "replay/INCAR").read_text()
    assert not (tmp_path / "replay/CHGCAR").exists()
    (ref / "POTCAR").write_text("different dataset")
    with pytest.raises(ValueError, match="POTCAR changed"):
        prepare_replay(operator, tmp_path / "wrong_dataset")


def test_padded_paw_predictions_are_not_silently_discarded(tmp_path):
    with pytest.raises(ValueError, match="padded"):
        write_paw(
            tmp_path / "bad",
            dict(
                d=np.ones((1, 2, 2)), q=np.zeros((1, 2, 2)), types=[0], channels=[[0]]
            ),
        )


def test_native_pot_geometry_can_wrap_atoms_but_not_move_them():
    reference = bulk("LiF", "rocksalt", a=4.03)
    wrapped = reference.copy()
    wrapped.positions[1] += reference.cell[0]
    check_pot_geometry(wrapped, reference)
    wrapped.positions[1, 0] += 0.001
    with pytest.raises(ValueError, match="structure differ"):
        check_pot_geometry(wrapped, reference)


def test_eigenval_precision_is_preserved_and_crosschecked(tmp_path):
    (tmp_path / "vasprun.xml").write_text("""<modeling>
      <kpoints><varray name="kpointlist"><v>0 0 0</v></varray></kpoints>
      <calculation><eigenvalues><array><set><set comment="spin 1">
      <set comment="kpoint 1"><r>1.2346 1</r><r>2.3457 0</r></set>
      </set></set></array></eigenvalues></calculation></modeling>""")
    text = "header\n" * 5 + "2 1 2\n\n0 0 0 1\n1 1.234567 1\n2 2.345678 0\n"
    (tmp_path / "EIGENVAL").write_text(text)
    np.testing.assert_allclose(
        read_bands(tmp_path)["eigenvalues_eV"], [[1.234567, 2.345678]], atol=0, rtol=0
    )
    (tmp_path / "EIGENVAL").write_text(text.replace("1.234567", "1.235567"))
    with pytest.raises(ValueError, match="printed precision"):
        read_bands(tmp_path)
