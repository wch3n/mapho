"""Fail before expensive reference calculations on incompatible UPF requests."""

import hashlib

import pytest

from mapho.qe.exchange import reference_spec


def pseudo(tmp_path, symbol, valence):
    path = tmp_path / f"{symbol}.upf"
    path.write_text(
        f'<UPF version="2.0.1"><PP_HEADER element="{symbol}" '
        f'z_valence="{valence}" pseudo_type="NC" functional="PBE" '
        'relativistic="scalar" number_of_proj="4"/>'
        "<PP_LOCAL/><PP_NONLOCAL/><PP_RHOATOM/></UPF>"
    )
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_lif_counts_lithium_semicore_and_checks_species_and_hash(tmp_path):
    pp = {s: pseudo(tmp_path, s, z) for s, z in [("Li", 3), ("F", 7)]}
    request = dict(
        numbers=[3, 9],
        pseudo_paths={s: p[0] for s, p in pp.items()},
        pseudo_sha256_by_element={s: p[1] for s, p in pp.items()},
        ecutwfc_ry=100,
    )
    _, _, nocc, ecut, rho = reference_spec(request)
    assert (nocc, ecut, rho) == (5, 100, 400)
    with pytest.raises(ValueError, match="species"):
        reference_spec({**request, "numbers": [14, 14]})
    with pytest.raises(ValueError, match="mismatch"):
        reference_spec(
            {**request, "pseudo_sha256_by_element": {"Li": "wrong", "F": pp["F"][1]}}
        )
    with pytest.raises(ValueError, match="doubly occupied"):
        reference_spec({**request, "numbers": [3, 3, 9]})
    with pytest.raises(ValueError, match="cutoffs"):
        reference_spec({**request, "ecutrho_ry": 160})


def test_legacy_silicon_reference_remains_compatible(tmp_path):
    path, digest = pseudo(tmp_path, "Si", 4)
    _, _, nocc, ecut, rho = reference_spec(
        dict(numbers=[14, 14], pseudo=path, pseudo_sha256=digest)
    )
    assert (nocc, ecut, rho) == (4, 40, 160)
