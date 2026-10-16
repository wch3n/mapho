"""Physical controls and failure checks for the MPI supercell references."""

import numpy as np
import pytest

from mapho.qe.supercell_reference import HARTREE_EV, read_spectrum, structure


def test_repeated_wavelength_control_is_the_same_distorted_crystal():
    short, _ = structure(4, "longitudinal")
    repeated, _ = structure(8, "longitudinal", harmonic=2)
    expected = short.repeat((2, 1, 1))

    def ordered(atoms):
        return sorted(
            (int(z), *np.round(p, 8)) for z, p in zip(atoms.numbers, atoms.positions)
        )

    np.testing.assert_allclose(ordered(expected), ordered(repeated), atol=1e-8)


def test_large_localized_perturbation_leaves_observation_neighborhood_unchanged():
    clean, _ = structure(8, "pristine")
    perturbed, info = structure(8, "localized")
    x = clean.positions[:, 0]
    distance_to_observation = np.minimum(x, info["length_angstrom"] - x)
    local = distance_to_observation <= 12.0
    np.testing.assert_allclose(
        perturbed.positions[local], clean.positions[local], atol=1e-12
    )
    assert np.max(np.abs(perturbed.positions[~local] - clean.positions[~local])) > 0.05


def test_size_and_reciprocal_scale_are_physical():
    for repeats, count in ((4, 32), (6, 48), (8, 64)):
        atoms, info = structure(repeats, "transverse")
        assert len(atoms) == count
        assert sum(atoms.numbers == 3) == sum(atoms.numbers == 9) == count // 2
        assert info["minimum_distance_angstrom"] > 1.8
        assert info["fundamental_q_inv_angstrom"] * info[
            "length_angstrom"
        ] == pytest.approx(2 * np.pi)


def test_xml_units_and_hybrid_outer_loop_are_checked(tmp_path):
    xml = tmp_path / "data-file-schema.xml"
    xml.write_text("""<q:espresso xmlns:q="urn:qe"><output>
      <convergence_info><scf_conv><convergence_achieved>true</convergence_achieved>
      <scf_error>1e-12</scf_error></scf_conv></convergence_info>
      <total_energy><etot>-10</etot></total_energy>
      <band_structure><ks_energies><k_point weight="2">0 0 0</k_point>
      <eigenvalues>-1 0.5</eigenvalues><occupations>1 0</occupations>
      </ks_energies></band_structure></output></q:espresso>""")
    log = tmp_path / "qe.log"
    log.write_text("convergence has been achieved\nJOB DONE.\n")
    result = read_spectrum(xml, log, hybrid=False, nocc=1)
    assert result["sampled_gap_eV"] == pytest.approx(1.5 * HARTREE_EV)
    assert result["weights"] == [1.0]
    with pytest.raises(RuntimeError, match="outer exchange"):
        read_spectrum(xml, log, hybrid=True, nocc=1)
    log.write_text("EXX self-consistency reached\nJOB DONE.\n")
    read_spectrum(xml, log, hybrid=True, nocc=1)
    xml.write_text(xml.read_text().replace(">true<", ">false<"))
    with pytest.raises(RuntimeError, match="XML charge convergence"):
        read_spectrum(xml, log, hybrid=True, nocc=1)
