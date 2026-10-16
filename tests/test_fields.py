import copy

import numpy as np
import pytest

from mapho.fields import FIELD_UNITS, FieldSample, from_qe_grid, to_qe_grid
from mapho.qe.backend import QEConfig, prepare_reference
from mapho.qe.upf import read_upf_metadata


def example():
    shape = (3, 4, 5)
    fields = {name: np.zeros(shape) for name in FIELD_UNITS}
    fields["density"][:] = 2 / 24
    fields["reference_density"][:] = 2 / 24
    fields["hartree"] = np.arange(60, dtype=float).reshape(shape) / 10
    fields["xc"][:] = -3
    fields["reference_hxc"][:] = 0.7
    return FieldSample(
        np.array([2]),
        np.zeros((1, 3)),
        np.diag([2.0, 3.0, 4.0]),
        fields,
        np.zeros((1, 3)),
        np.ones(1),
        np.array([[-2.0, 1.0]]),
        np.array([[2.0, 0.0]]),
        {
            "schema": "mlham.scalar_fields.v1",
            "field_units": FIELD_UNITS,
            "length_unit": "angstrom",
            "energy_unit": "eV",
            "grid_convention": "fractional_ijk_endpoint_excluded",
            "grid_origin_fractional": [0, 0, 0],
            "pbc": [True, True, True],
            "nspin": 1,
            "potential_gauge": "test",
            "reference_definition": "test",
            "nelect": 2,
            "pseudopotentials": {"He": {"sha256": "a" * 64}},
        },
    )


def test_grid_order_and_coordinates():
    # Known linear index i + nx*(j + ny*k), with unequal axis lengths.
    nx, ny, nz = 3, 4, 5
    flat = np.arange(nx * ny * nz).reshape(-1, 1)
    grid = from_qe_grid(flat, (nx, ny, nz))
    assert grid[1, 2, 3] == 1 + nx * (2 + ny * 3)
    np.testing.assert_array_equal(to_qe_grid(grid), flat)
    with pytest.raises(ValueError, match="scalar"):
        from_qe_grid(np.zeros((60, 2)), (3, 4, 5))


def test_fields_serialization_and_offset(tmp_path):
    sample = example()
    sample.save(tmp_path / "fields.npz")
    loaded = FieldSample.load(tmp_path / "fields.npz")
    centered, offset = loaded.residual()
    assert abs(centered.mean()) < 1e-14
    np.testing.assert_allclose(
        centered + offset + loaded.fields["reference_hxc"], loaded.hxc
    )
    np.testing.assert_array_equal(loaded.cell, sample.cell)
    with pytest.raises(FileExistsError):
        loaded.save(tmp_path / "fields.npz")


@pytest.mark.parametrize(
    "corruption", ["units", "density", "weights", "occupation", "spin", "nan"]
)
def test_inconsistent_fields_rejected(corruption):
    sample = copy.deepcopy(example())
    if corruption == "units":
        sample.metadata["field_units"]["hartree"] = "Ry"
    elif corruption == "density":
        sample.fields["density"] *= 2
    elif corruption == "weights":
        sample.kweights *= 2
    elif corruption == "occupation":
        sample.occupations[0, 0] = 1
    elif corruption == "spin":
        sample.metadata["nspin"] = 2
    else:
        sample.eigenvalues[0, 0] = np.nan
    with pytest.raises(ValueError):
        sample.validate()


def pseudo(tmp_path, **attributes):
    attrs = {
        "element": "He",
        "pseudo_type": "NC",
        "functional": "PBE",
        "z_valence": "2",
        "number_of_proj": "1",
    }
    attrs.update(attributes)
    path = tmp_path / "He.upf"
    text = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    path.write_text(
        f'<UPF version="2.0.1"><PP_HEADER {text}/>'
        "<PP_LOCAL/><PP_NONLOCAL/><PP_RHOATOM/></UPF>"
    )
    return path


@pytest.mark.parametrize(
    "attributes",
    [
        {"pseudo_type": "US"},
        {"is_paw": "T"},
        {"has_so": "T"},
        {"functional": "LDA"},
    ],
)
def test_unsupported_pseudopotentials_rejected(tmp_path, attributes):
    with pytest.raises(ValueError):
        read_upf_metadata(pseudo(tmp_path, **attributes))


def test_preparation_checks_and_preserves_pseudo(tmp_path):
    from ase import Atoms

    atoms = Atoms("He", cell=[4, 4, 4], pbc=True)
    path = pseudo(tmp_path)
    request = prepare_reference(atoms, {"He": path}, tmp_path / "run", QEConfig(nbnd=2))
    assert request.exists()
    assert (request.parent / "pseudos/He.upf").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="element mismatch"):
        prepare_reference(
            Atoms("Ne", cell=[4, 4, 4], pbc=True), {"Ne": path}, tmp_path / "wrong"
        )
    atoms.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        prepare_reference(atoms, {"He": path}, tmp_path / "nonperiodic")


@pytest.mark.parametrize(
    "config",
    [
        QEConfig(ecutwfc_ry=-1),
        QEConfig(kmesh=(0, 2, 2)),
        QEConfig(nbnd=1.5),
        QEConfig(ecutrho_ry=20),
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        config.validate()


def test_atomic_reference_has_no_scf_labels(tmp_path):
    from mapho.fields import AtomicReference, load_operator_reference
    from mapho.qe.backend import solve

    sample = example()
    metadata = copy.deepcopy(sample.metadata)
    metadata.update(
        schema="mlham.atomic_reference.v1",
        functional="PBE",
        pseudo_paths={"He": "pseudos/He.upf"},
        qe_config={"nbnd": 2},
    )
    fields = {n: sample.fields[n] for n in ("reference_density", "reference_hxc")}
    metadata["field_units"] = {n: FIELD_UNITS[n] for n in fields}
    ref = AtomicReference(
        sample.numbers, sample.positions, sample.cell, fields, sample.kpoints, metadata
    )
    path = tmp_path / "atomic.npz"
    ref.save(path)
    loaded = load_operator_reference(path)
    assert isinstance(loaded, AtomicReference)
    assert not hasattr(loaded, "eigenvalues")
    np.testing.assert_array_equal(
        loaded.fields["reference_hxc"], ref.fields["reference_hxc"]
    )
    with pytest.raises(ValueError, match="explicit predicted"):
        solve(path, tmp_path / "invalid")
    ref.fields["hartree"] = np.zeros(ref.shape)
    with pytest.raises(ValueError, match="only initial"):
        ref.validate()


@pytest.mark.parametrize("points", [[], [0, 0, 0], [[0, 0]], [[0, float("nan"), 0]]])
def test_invalid_band_kpoints_rejected_before_output(tmp_path, points):
    from mapho.qe.backend import solve

    path = tmp_path / "fields.npz"
    example().save(path)
    with pytest.raises(ValueError, match="Band k points"):
        solve(path, tmp_path / "invalid", kpoints=points)
    assert not (tmp_path / "invalid").exists()


def test_band_request_preserves_path_and_potential(tmp_path, monkeypatch):
    import json
    from mapho.qe import backend as qe

    sample = example()
    sample.metadata.update(pseudo_paths={"He": "pseudos/He.upf"}, qe_config={"nbnd": 2})
    path = tmp_path / "fields.npz"
    sample.save(path)
    points = np.array([[0.5, 0, 0.5], [0.25, 0, 0.25], [0, 0, 0], [0.5, 0.5, 0.5]])

    def launch(request_path):
        request = json.loads(request_path.read_text())
        assert request["mode"] == "bands"
        np.testing.assert_array_equal(request["kpoints"], points)
        np.testing.assert_allclose(
            np.load(request_path.parent / "input_hxc_eV.npy"), sample.hxc + 0.2
        )
        # Request/serialization test only; real QE comparisons validate spectra.
        np.savez(request_path.parent / "bands.npz", kpoints_fractional=points)

    monkeypatch.setattr(qe, "_launch", launch)
    result = qe.solve(path, tmp_path / "bands", constant_eV=0.2, kpoints=points)
    np.testing.assert_array_equal(result["kpoints_fractional"], points)
    assert "density" not in result
