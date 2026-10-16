"""Synthetic transport and independent factorization; no licensed data needed."""

import json

import h5py
import numpy as np
import pytest

from mapho.vasp.ace import (
    SCHEMA,
    apply_ace,
    prepare_hybrid_replay,
    read_ace,
    read_exchange_samples,
    write_ace,
    write_exchange_samples,
)
from mapho.vasp.exchange import build_ace_from_samples, compare_native_factors
from mapho.file_utils import sha
from test_vasp_hybrid_pilot import hybrid_components


def samples():
    rng = np.random.default_rng(77)
    orbitals, actions = [], []
    for ng in (9, 11):
        # PAW-normalized columns need not be Euclidean orthonormal.
        phi = rng.normal(size=(ng, 3)) + 1j * rng.normal(size=(ng, 3))
        a = rng.normal(size=(ng, ng)) + 1j * rng.normal(size=(ng, ng))
        k = -(a @ a.conj().T + np.eye(ng)) * 0.03
        orbitals.append(phi)
        actions.append(k @ phi)
    return dict(
        dims=np.array([12, 3, 2, 1]),
        grid=np.array([4, 5, 6]),
        cell_A=np.array([[4.0, 0.1, 0.2], [0.0, 5.0, 0.3], [0.0, 0.0, 6.0]]),
        fraction=0.25,
        kpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0]]),
        indices=[np.arange(1, 10), np.arange(20, 31)],
        orbitals=orbitals,
        actions=actions,
    )


def test_independent_factorization_and_unitary_freedom(tmp_path):
    source = tmp_path / "samples.bin"
    ours = tmp_path / "ours.bin"
    rotated = tmp_path / "rotated.bin"
    original = samples()
    write_exchange_samples(source, original)
    restored = read_exchange_samples(source)
    for key in ("dims", "grid", "cell_A", "kpoints"):
        np.testing.assert_array_equal(original[key], restored[key])
    for key in ("indices", "orbitals", "actions"):
        for before, after in zip(original[key], restored[key]):
            np.testing.assert_array_equal(before, after)
    result = build_ace_from_samples(source, ours)
    assert not result["native_ace_used_for_factorization"]
    assert max(result["action_max_error_eV"]) < 1e-12
    archive = read_ace(ours)
    for k, phi in enumerate(original["orbitals"]):
        np.testing.assert_allclose(
            apply_ace(archive, k, phi), original["actions"][k], atol=1e-12
        )
        matrix = apply_ace(archive, k, np.eye(len(phi)))
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)
        assert np.linalg.eigvalsh(matrix).max() < 1e-12
    archive["factors"] = [x[:, ::-1] * 1j for x in archive["factors"]]
    write_ace(rotated, archive)
    assert compare_native_factors(source, ours, rotated)["passed"]
    write_ace(tmp_path / "half.bin", archive, scale=0.5)
    assert not compare_native_factors(source, ours, tmp_path / "half.bin")["passed"]
    half = read_ace(tmp_path / "half.bin")
    np.testing.assert_allclose(
        apply_ace(half, 0, original["orbitals"][0]),
        0.5 * original["actions"][0],
        atol=1e-12,
    )


@pytest.mark.parametrize("damage", ["short", "trailing", "header"])
def test_binary_damage_is_rejected(tmp_path, damage):
    path = tmp_path / "samples.bin"
    write_exchange_samples(path, samples())
    raw = path.read_bytes()
    path.write_bytes(
        {"short": raw[:-1], "trailing": raw + b"!", "header": b"!" + raw[1:]}[damage]
    )
    with pytest.raises(ValueError):
        read_exchange_samples(path)


@pytest.mark.parametrize(
    "damage", ["count", "mapping", "nonfinite", "shape", "spin", "cell"]
)
def test_invalid_archive_is_rejected_before_writing(tmp_path, damage):
    archive = samples()
    if damage == "count":
        archive["actions"].pop()
    elif damage == "mapping":
        archive["indices"][0][1] = archive["indices"][0][0]
    elif damage == "nonfinite":
        archive["actions"][0][0, 0] = np.nan
    elif damage == "shape":
        archive["orbitals"][0] = archive["orbitals"][0][:, :2]
    elif damage == "spin":
        archive["dims"][-1] = 2
    elif damage == "cell":
        archive["cell_A"][0] *= -1
    path = tmp_path / "bad.bin"
    with pytest.raises(ValueError):
        write_exchange_samples(path, archive)
    assert not path.exists()


def test_mismatched_reciprocal_order_is_rejected(tmp_path):
    source = tmp_path / "samples.bin"
    ours = tmp_path / "ours.bin"
    changed = tmp_path / "changed.bin"
    write_exchange_samples(source, samples())
    build_ace_from_samples(source, ours)
    archive = read_ace(ours)
    archive["indices"][0] = archive["indices"][0][::-1]
    write_ace(changed, archive)
    with pytest.raises(ValueError, match="ordering"):
        compare_native_factors(source, ours, changed)


def test_nonhermitian_or_positive_actions_are_rejected(tmp_path):
    for kind in ("positive", "nonhermitian"):
        archive = samples()
        archive["actions"] = [
            w * (-1 if kind == "positive" else 1j) for w in archive["actions"]
        ]
        path = tmp_path / f"{kind}.bin"
        write_exchange_samples(path, archive)
        with pytest.raises(ValueError):
            build_ace_from_samples(path, tmp_path / f"{kind}.ace")


def hybrid_archive(tmp_path):
    components = hybrid_components(tmp_path)
    with h5py.File(components) as f:
        component_meta = json.loads(f.attrs["metadata_json"])
    source = tmp_path / "samples.bin"
    factors = tmp_path / "factors.bin"
    write_exchange_samples(source, samples())
    build_ace_from_samples(source, factors)
    meta = dict(
        components=str(components),
        components_sha256=sha(components),
        ace_file=str(factors),
        ace_sha256=sha(factors),
        reference_manifest=component_meta["reference_manifest"],
        representation="synthetic finite-rank PBE0 operator",
    )
    path = tmp_path / "operator.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = SCHEMA
        f.attrs["metadata_json"] = json.dumps(meta)
    return path


def test_replay_is_frozen_and_reads_no_reference_wavefunctions(tmp_path):
    operator = hybrid_archive(tmp_path)
    folder = tmp_path / "replay"
    m = prepare_hybrid_replay(operator, folder)
    s = m["settings"]
    assert s["LHFCALC"] and s["LFOCKACE"] and s["ALGO"] == "Normal"
    assert (s["ICHARG"], s["ISTART"], s["NELMDL"]) == (14, 0, 0)
    assert not s["LWAVE"] and not s["LCHARG"]
    assert "TIME" not in s
    assert (folder / "MAPHO_ACE_IMPORT").exists() and (folder / "MAPHO_IMPORT").exists()
    assert not any((folder / n).exists() for n in ("WAVECAR", "CHGCAR"))
    assert m["complete_operator_components"] and not m["exact_arbitrary_fock_action"]
    assert "omitted_components" not in m


def test_replay_detects_tampering_and_invalid_scale(tmp_path):
    operator = hybrid_archive(tmp_path)
    with pytest.raises(ValueError, match="scale"):
        prepare_hybrid_replay(operator, tmp_path / "bad_scale", ace_scale=-1)
    assert not (tmp_path / "bad_scale").exists()
    with (tmp_path / "factors.bin").open("ab") as f:
        f.write(b"!")
    with pytest.raises(ValueError, match="changed"):
        prepare_hybrid_replay(operator, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
