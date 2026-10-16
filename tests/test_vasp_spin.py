from io import StringIO
import json

from ase.build import bulk
from ase.io import write
import h5py
import numpy as np
import pytest

from mapho.vasp.paw import (
    SPIN_SCHEMA,
    prepare_reference,
    prepare_replay,
    read_bands,
    read_paw,
    read_pot,
    write_paw,
    write_pot,
)
from mapho.file_utils import sha


def spin_template():
    atoms = bulk("Fe", "bcc", a=2.86)
    out = StringIO()
    write(out, atoms, format="vasp", direct=True, vasp5=True)
    aug = "augmentation occupancies   1  1\n0.125\n"
    return out.getvalue(), [aug + " 2.5\n", aug]


def test_two_pot_grids_are_spin_channels_and_both_are_preserved(tmp_path):
    header, suffix = spin_template()
    x, y, z = np.indices((2, 3, 4))
    up = 2 + 0.1 * x + y + 10 * z
    down = -3 + 0.2 * x + 2 * y - 10 * z
    path = tmp_path / "POT"
    write_pot(path, np.stack([up, down]), header, suffix)
    atoms, field, h, s = read_pot(path)
    assert len(atoms) == 1
    np.testing.assert_allclose(field, np.stack([up, down]), atol=1e-12, rtol=0)
    assert s == suffix
    # Removing the second grid must not silently produce an ISPIN=1 field.
    text = path.read_text()
    path.write_text(text[: text.rindex("2 3 4")])
    with pytest.raises(ValueError, match="Truncated"):
        read_pot(path)


def test_spin_matrices_have_one_shared_overlap(tmp_path):
    d = np.array([[[[1.0, 0.2], [0.2, 2.0]]], [[[-1.0, 0.4], [0.4, 3.0]]]])
    q = np.array([[[0.2, 0.01], [0.01, 0.1]]])
    paw = dict(d=d, q=q, types=np.array([0]), channels=[[0, 0]])
    path = tmp_path / "MAPHO_PAW.dat"
    write_paw(path, paw)
    assert path.read_text().startswith("MAPHO_PAW_V2\n2 1 1 2\n")
    result = read_paw(path)
    np.testing.assert_allclose(result["d"], d, atol=0, rtol=0)
    np.testing.assert_allclose(result["q"], q, atol=0, rtol=0)
    path.write_text(path.read_text().replace("2 1 1 2", "2 1 1 4", 1))
    with pytest.raises(ValueError, match="spin dimensions"):
        read_paw(path)


def spin_operator(tmp_path):
    header, suffix = spin_template()
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "POSCAR").write_text(header)
    (ref / "POTCAR").write_text("synthetic test identity; not a real PAW dataset")
    (ref / "KPOINTS").write_text("synthetic k mesh")
    metadata = dict(
        reference=str(ref),
        reference_manifest=dict(
            settings=dict(ISPIN=2, MAGMOM="2.5"),
            inputs={n: sha(ref / n) for n in ("POSCAR", "POTCAR", "KPOINTS")},
        ),
    )
    operator = tmp_path / "operator.h5"
    with h5py.File(operator, "w") as f:
        f.attrs["schema"] = SPIN_SCHEMA
        f.attrs["metadata_json"] = json.dumps(metadata)
        f["potential_eV"] = np.stack([np.full((2, 3, 4), 3.0), np.full((2, 3, 4), 5.0)])
        f["paw/d"] = np.array([[[[1.0]]], [[[2.0]]]])
        f["paw/q"] = np.array([[[0.2]]])
        f["paw/types"] = [0]
        f["paw"].attrs["channels_json"] = "[[0]]"
        f["native_pot_header"] = header
        f.create_dataset("native_pot_suffixes", data=suffix, dtype=h5py.string_dtype())
    return operator


@pytest.mark.parametrize("schema", [SPIN_SCHEMA, "mlham.vasp_paw_operator.v2"])
def test_independent_spin_shifts_obey_generalized_eigenproblem(tmp_path, schema):
    op = spin_operator(tmp_path)
    with h5py.File(op, "r+") as f:
        f.attrs["schema"] = schema
    shifts = np.array([0.125, -0.375])
    prepare_replay(op, tmp_path / "replay", shift_eV=shifts)
    _, v, _, _ = read_pot(tmp_path / "replay/POT")
    paw = read_paw(tmp_path / "replay/MAPHO_PAW.dat")
    p = np.array([[0.3], [0.2], [0.1]])
    overlap = np.eye(3) + p @ paw["q"][0] @ p.T
    inv_l = np.linalg.inv(np.linalg.cholesky(overlap))
    for spin, (v0, d0) in enumerate([(3.0, 1.0), (5.0, 2.0)]):
        old = np.diag(np.arange(3) + v0) + p @ np.array([[d0]]) @ p.T
        new = np.diag(np.arange(3) + v[spin, 0, 0, 0]) + p @ paw["d"][spin, 0] @ p.T
        eig = lambda h: np.linalg.eigvalsh(inv_l @ h @ inv_l.T)
        np.testing.assert_allclose(eig(new) - eig(old), shifts[spin], atol=1e-14)
    incar = (tmp_path / "replay/INCAR").read_text()
    assert "NGXF = 2" in incar and "NGYF = 3" in incar and "NGZF = 4" in incar
    assert "ISPIN = 2" in incar and "ICHARG = 14" in incar


def test_spin_reversal_changes_both_field_and_d(tmp_path):
    op = spin_operator(tmp_path)
    prepare_replay(op, tmp_path / "reversed", reverse_spins=True)
    _, v, _, _ = read_pot(tmp_path / "reversed/POT")
    paw = read_paw(tmp_path / "reversed/MAPHO_PAW.dat")
    np.testing.assert_allclose(v[:, 0, 0, 0], [5.0, 3.0])
    np.testing.assert_allclose(paw["d"][:, 0, 0, 0], [2.0, 1.0])
    assert "MAGMOM = -2.5" in (tmp_path / "reversed/INCAR").read_text()


def test_spin_eigenval_order_occupations_and_moment(tmp_path):
    (tmp_path / "vasprun.xml").write_text("""<modeling>
      <kpoints><varray name="kpointlist"><v>0 0 0</v></varray><varray name="weights"><v>1</v></varray></kpoints>
      <calculation><eigenvalues><array><set>
      <set comment="spin 1"><set><r>1.2346 1</r><r>2.3457 0</r></set></set>
      <set comment="spin 2"><set><r>3.4568 .5</r><r>4.5679 0</r></set></set>
      </set></array></eigenvalues></calculation></modeling>""")
    (tmp_path / "EIGENVAL").write_text(
        "header\n" * 5
        + "2 1 2\n\n0 0 0 1\n1 1.234567 3.456789 1 .5\n2 2.345678 4.567891 0 0\n"
    )
    b = read_bands(tmp_path)
    assert b["nspin"] == 2 and b["eigenvalues_eV"].shape == (2, 1, 2)
    np.testing.assert_allclose(
        b["eigenvalues_eV"][:, 0, :],
        [[1.234567, 2.345678], [3.456789, 4.567891]],
        atol=0,
        rtol=0,
    )
    assert b["spin_moment_muB"] == 0.5


@pytest.mark.parametrize(
    "ispin,moments", [(1, [2.5]), (2, [[0.0, 0.0, 2.5]]), (4, [2.5])]
)
def test_invalid_spin_modes_rejected_before_preparation(tmp_path, ispin, moments):
    with pytest.raises(ValueError):
        prepare_reference(
            tmp_path / "invalid",
            bulk("Fe", "bcc", a=2.86),
            {},
            ispin=ispin,
            magmoms=moments,
        )
