"""A partial hybrid export must never masquerade as a complete PAW operator."""

import json

from ase.build import bulk
from ase.io import write
import h5py
import numpy as np
import pytest

from mapho.vasp.paw import (
    COMPONENT_SCHEMA,
    SCHEMA,
    export_operator,
    hybrid_reference,
    prepare_replay,
    read_paw,
    read_pot,
    sha,
    write_paw,
)


@pytest.mark.parametrize(
    "functional,flag", [("PBE0", False), ("PBE", ".TRUE."), ("PBE", True)]
)
def test_hybrid_export_cannot_claim_complete_hamiltonian(tmp_path, functional, flag):
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(functional=functional, settings={"LHFCALC": flag}))
    )
    with pytest.raises(ValueError, match="nonlocal Fock"):
        export_operator(tmp_path, tmp_path / "operator.h5")
    assert not (tmp_path / "operator.h5").exists()
    assert not hybrid_reference(dict(functional="PBE", settings={"LHFCALC": ".FALSE."}))


def hybrid_components(tmp_path, schema=COMPONENT_SCHEMA):
    ref = tmp_path / "reference"
    ref.mkdir()
    write(
        ref / "POSCAR",
        bulk("Si", "diamond", a=5.43),
        format="vasp",
        direct=True,
        vasp5=True,
    )
    (ref / "POTCAR").write_text("synthetic identity, never passed to VASP")
    (ref / "KPOINTS").write_text("synthetic mesh")
    meta = dict(
        reference=str(ref),
        reference_manifest=dict(
            functional="PBE0",
            settings=dict(
                ISPIN=1,
                LHFCALC=True,
                AEXX=0.25,
                HFSCREEN=0.0,
                HFRCUT=0.0,
                LFOCKACE=False,
                ALGO="Damped",
                TIME=0.4,
            ),
            inputs={n: sha(ref / n) for n in ("POSCAR", "POTCAR", "KPOINTS")},
        ),
    )
    path = tmp_path / "components.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = schema
        f.attrs["metadata_json"] = json.dumps(meta)
        f["potential_eV"] = np.arange(24).reshape(2, 3, 4) * 0.1
        f["paw/d"] = np.array([[[1.0]], [[2.0]]])
        f["paw/q"] = np.array([[[0.2]], [[0.1]]])
        f["paw/types"] = [0, 0]
        f["paw"].attrs["channels_json"] = "[[0]]"
        f["native_pot_header"] = (ref / "POSCAR").read_text()
        f["native_pot_suffix"] = (
            "augmentation occupancies   1  1\n0\naugmentation occupancies   2  1\n0\n"
        )
    return path


def test_hybrid_diagnostic_is_explicit_and_preserves_exact_components(tmp_path):
    source = hybrid_components(tmp_path)
    with pytest.raises(ValueError, match="full Hamiltonian"):
        prepare_replay(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    out = tmp_path / "diagnostic"
    m = prepare_replay(source, out, omit_fock=True)
    assert m["settings"]["LHFCALC"] is False
    assert m["settings"]["ICHARG"] == 14
    assert m["settings"]["ALGO"] == "Exact"
    assert "AEXX" not in m["settings"]
    assert not m["complete_hamiltonian"]
    assert m["omitted_components"] == ["nonlocal Fock operator"]
    with h5py.File(source) as f:
        np.testing.assert_allclose(
            read_pot(out / "POT")[1], f["potential_eV"][:], atol=1e-14
        )
        np.testing.assert_array_equal(
            read_paw(out / "MLHAM_PAW.dat")["d"], f["paw/d"][:]
        )
    assert not (out / "WAVECAR").exists()


def test_incorrectly_labelled_legacy_hybrid_cannot_bypass_guard(tmp_path):
    source = hybrid_components(tmp_path, SCHEMA)
    for omit in (False, True):
        with pytest.raises(ValueError, match="full Hamiltonian"):
            prepare_replay(source, tmp_path / "invalid", omit_fock=omit)


def test_hybrid_roundoff_tolerance_preserves_raw_values_and_leaves_q_strict(tmp_path):
    path = tmp_path / "paw.dat"
    write_paw(
        path,
        dict(
            d=np.array([[[1.0, 0.2], [0.2, 2.0]]]),
            q=np.zeros((1, 2, 2)),
            types=np.array([0]),
            channels=[[0, 0]],
        ),
    )
    lines = path.read_text().splitlines()
    original = lines.copy()
    lines[6] = "0.20000014 0.0"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="symmetric"):
        read_paw(path)
    raw = read_paw(path, d_symmetry_tolerance_eV=1e-6)
    assert raw["d"][0, 1, 0] == 0.20000014
    with pytest.raises(ValueError, match="symmetric"):
        write_paw(tmp_path / "cannot_replay_raw", raw)
    lines = original
    lines[6] = "0.2 0.00000014"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="symmetric"):
        read_paw(path, d_symmetry_tolerance_eV=1e-6)


@pytest.mark.parametrize("noise", [1.4e-7, 1.1e-6])
def test_only_small_native_hybrid_asymmetry_can_be_projected(tmp_path, noise):
    source = hybrid_components(tmp_path)
    raw = np.tile(np.array([[1.0, 0.2], [0.2 + noise, 2.0]]), (2, 1, 1))
    with h5py.File(source, "r+") as f:
        del f["paw/d"]
        del f["paw/q"]
        f["paw/d"] = raw
        f["paw/q"] = np.zeros_like(raw)
        f["paw"].attrs["channels_json"] = "[[0, 0]]"
    digest = sha(source)
    if noise > 1e-6:
        with pytest.raises(ValueError, match="asymmetry exceeds"):
            prepare_replay(source, tmp_path / "diagnostic", omit_fock=True)
    else:
        m = prepare_replay(source, tmp_path / "diagnostic", omit_fock=True)
        np.testing.assert_allclose(
            read_paw(tmp_path / "diagnostic/MLHAM_PAW.dat")["d"],
            0.5 * (raw + raw.swapaxes(-1, -2)),
            rtol=0,
            atol=1e-15,
        )
        assert m["d_hermitization_max_eV"] == pytest.approx(noise / 2)
    assert sha(source) == digest
