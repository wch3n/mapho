"""Prepare and run MPI QE references for a longitudinal LiF size experiment.

This produces native SCF saves and checked spectra. It deliberately does not
label them as the eight-band exchange-action NPZ format used by the old pilot.
No QEpy or PyTorch imports are needed in the MPI launcher.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

from .upf import read_upf_metadata
from .files import save_directory

HARTREE_EV = 27.211386245988034


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def structure(repeats, pattern, harmonic=1):
    """Fixed-volume insulating geometries; no primitive-cell random strain."""
    if repeats not in (4, 6, 8) or harmonic not in (1, 2):
        raise ValueError("Unsupported cell length or harmonic")
    atoms = bulk("LiF", "rocksalt", a=4.03, cubic=True).repeat((repeats, 1, 1))
    x = atoms.positions[:, 0].copy()
    length = atoms.cell[0, 0]
    phase = 2 * np.pi * harmonic * x / length
    sign = np.where(atoms.numbers == 3, 1.0, -1.0)
    displacement = np.zeros_like(atoms.positions)
    if pattern == "longitudinal":
        displacement[:, 0] = 0.04 * sign * np.sin(phase + 0.37)
    elif pattern == "transverse":
        displacement[:, 1] = 0.06 * sign * np.sin(phase + 0.71)
    elif pattern == "mixed":
        displacement[:, 0] = 0.035 * sign * np.sin(phase + 1.13)
        displacement[:, 1] = 0.05 * np.sin(2 * phase + 0.29)
    elif pattern == "localized":
        # Periodic compact bump, identically zero away from its support. In the
        # 32 A cell its atoms lie beyond 12 A from the x=0 observation plane.
        distance = (x - length / 2 + length / 2) % length - length / 2
        inside = np.abs(distance) < 2.0
        displacement[inside, 0] = (
            0.06 * sign[inside] * np.cos(np.pi * distance[inside] / 4) ** 2
        )
    elif pattern != "pristine":
        raise ValueError("Unknown distortion")
    atoms.positions += displacement
    atoms.wrap()
    nearest = atoms.get_all_distances(mic=True)
    np.fill_diagonal(nearest, np.inf)
    if nearest.min() < 1.8:
        raise ValueError("Distortion creates an unreasonably short contact")
    info = dict(
        repeats=[repeats, 1, 1],
        pattern=pattern,
        harmonic=harmonic,
        length_angstrom=float(length),
        wavelength_angstrom=float(length / harmonic),
        fundamental_q_inv_angstrom=float(2 * np.pi / length),
        perturbation_q_inv_angstrom=float(2 * np.pi * harmonic / length),
        maximum_displacement_angstrom=float(np.linalg.norm(displacement, axis=1).max()),
        minimum_distance_angstrom=float(nearest.min()),
    )
    return atoms, info


def write_case(
    root,
    name,
    atoms,
    info,
    pseudos,
    *,
    mesh=(1, 2, 2),
    nbnd=None,
    conv_thr=1e-9,
    restart_pbe=True,
):
    out = root / "cases" / name
    out.mkdir(parents=True)
    write(out / "structure.extxyz", atoms)
    nelect = sum(pseudos[s]["z_valence"] for s in atoms.get_chemical_symbols())
    nocc = int(round(nelect / 2))
    if abs(nelect - 2 * nocc) > 1e-10:
        raise ValueError("Need a neutral closed-shell structure")
    nbnd = nocc + 8 if nbnd is None else nbnd
    if nbnd <= nocc:
        raise ValueError("Need unoccupied bands to check the insulating gap")
    case = dict(
        name=name,
        natoms=len(atoms),
        nocc=nocc,
        nbnd=nbnd,
        nelect=nelect,
        kmesh=list(mesh),
        info=info,
        inputs={},
    )
    for functional in ("PBE", "PBE0"):
        folder = out / functional
        folder.mkdir()
        hybrid = functional == "PBE0"
        system = dict(
            ecutwfc=100.0,
            ecutrho=400.0,
            nbnd=nbnd,
            input_dft=functional,
            occupations="fixed",
            nspin=1,
            nosym=True,
            noinv=True,
        )
        if hybrid:
            system.update(
                exx_fraction=0.25,
                screening_parameter=0.0,
                nqx1=mesh[0],
                nqx2=mesh[1],
                nqx3=mesh[2],
                exxdiv_treatment="gygi-baldereschi",
                x_gamma_extrapolation=False,
                ace=True,
            )
        restart = hybrid and restart_pbe
        params = dict(
            control=dict(
                calculation="scf",
                prefix="mapho",
                outdir=str(folder / "scratch"),
                pseudo_dir=str(root / "pseudos"),
                disk_io="low",
                max_seconds=6000 if hybrid else 900,
            ),
            system=system,
            electrons=dict(
                conv_thr=conv_thr,
                electron_maxstep=150,
                exx_maxstep=80,
                mixing_beta=0.4,
                diago_full_acc=True,
                startingpot="file" if restart else "atomic",
                startingwfc="file" if restart else "atomic+random",
            ),
        )
        write(
            folder / "qe.in",
            atoms,
            format="espresso-in",
            input_data=params,
            pseudopotentials={s: f"{s}.upf" for s in pseudos},
            kpts=mesh,
        )
        case["inputs"][functional] = dict(
            path=str(folder / "qe.in"),
            sha256=digest(folder / "qe.in"),
            restart_from_pbe=restart,
        )
    return case


def prepare(root, pseudo_dir, legacy_sample):
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("Dataset manifest already exists")
    (root / "pseudos").mkdir()
    pseudos = {}
    for symbol in ("Li", "F"):
        src = Path(pseudo_dir) / f"{symbol}_ONCV_PBE-1.2.upf"
        dst = root / "pseudos" / f"{symbol}.upf"
        shutil.copyfile(src, dst)
        pseudos[symbol] = read_upf_metadata(dst)
    cases = []
    for repeats in (4, 6, 8):
        for pattern in ("pristine", "longitudinal", "transverse", "mixed", "localized"):
            atoms, info = structure(repeats, pattern)
            name = f"li_f_{repeats}x1x1_{pattern}"
            cases.append(write_case(root, name, atoms, info, pseudos))
    atoms, info = structure(8, "longitudinal", harmonic=2)
    info["role"] = "same 16.12 A wavelength as 4x1x1 longitudinal, repeated twice"
    cases.append(write_case(root, "li_f_8x1x1_longitudinal_m2", atoms, info, pseudos))
    old = Path(legacy_sample).resolve()
    r = json.loads((old / "PBE0" / "request.json").read_text())
    atoms = Atoms(r["numbers"], positions=r["positions"], cell=r["cell"], pbc=True)
    pilot = write_case(
        root,
        "primitive_validation",
        atoms,
        {"role": "cross-version validation against archived QEpy 7.2"},
        pseudos,
        mesh=(2, 2, 2),
        nbnd=8,
        conv_thr=1e-10,
        restart_pbe=False,
    )
    pilot["legacy_sample"] = str(old)
    manifest = dict(
        schema="mapho.longitudinal_supercell_references.v1",
        cases=cases,
        pilot=pilot,
        pseudopotentials=pseudos,
        source_sha256=digest(__file__),
        scope="Fixed-geometry longitudinal size pilot, not a converged 3D large-cell benchmark or a completed ML comparison.",
        settings=dict(
            ecutwfc_ry=100,
            ecutrho_ry=400,
            kmesh=[1, 2, 2],
            exx_fraction=0.25,
            screening_bohr_inverse=0,
            exxdiv_treatment="gygi-baldereschi",
            x_gamma_extrapolation=False,
        ),
        native_outputs=["data-file-schema.xml", "charge-density.dat", "wfc*.dat"],
        exchange_action_export=False,
        notes=[
            "Lengths 16.12, 24.18, 32.24 A; transverse dimensions 4.03 A.",
            "Shortest reciprocal modes 0.390, 0.260, 0.195 inverse A.",
            "Pristine and repeated-wavelength cases separate folding/sampling effects from new spatial variation.",
            "The fixed k mesh is a feasibility setting, not a k-point convergence certificate.",
            "No training/test assignment or model selection has been made from these new labels.",
        ],
    )
    save_json(root / "manifest.json", manifest)
    shutil.copyfile(__file__, root / "runner_snapshot.py")
    return manifest


def read_spectrum(xml_path, log_path, *, hybrid, nocc):
    """Require charge and outer-exchange convergence, with Hartree XML units."""
    log = Path(log_path).read_text(errors="replace")
    if "JOB DONE." not in log or "convergence NOT achieved" in log:
        raise RuntimeError("QE did not complete a converged SCF")
    if hybrid and "EXX self-consistency reached" not in log:
        raise RuntimeError("PBE0 outer exchange loop did not converge")
    root = ET.parse(xml_path).getroot()
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    output = root.find("output")
    if (
        output is None
        or output.findtext("convergence_info/scf_conv/convergence_achieved") != "true"
    ):
        raise RuntimeError("XML charge convergence missing")
    ks = output.findall("band_structure/ks_energies")
    eigen = (
        np.array([np.fromstring(k.findtext("eigenvalues"), sep=" ") for k in ks])
        * HARTREE_EV
    )
    kpoints = np.array([np.fromstring(k.findtext("k_point"), sep=" ") for k in ks])
    weights = np.array([float(k.find("k_point").attrib["weight"]) for k in ks])
    occupations = np.array(
        [np.fromstring(k.findtext("occupations"), sep=" ") for k in ks]
    )
    if eigen.ndim != 2 or eigen.shape[1] <= nocc or not np.isfinite(eigen).all():
        raise RuntimeError("Invalid exported spectrum")
    if not np.allclose(occupations[:, :nocc], 1.0, atol=1e-7) or np.any(
        occupations[:, nocc:] > 1e-7
    ):
        raise RuntimeError("Unexpected fixed occupations in native XML")
    gap = float(eigen[:, nocc:].min() - eigen[:, :nocc].max())
    if gap <= 0:
        raise RuntimeError("Reference is not insulating")
    return dict(
        eigenvalues_eV=eigen.tolist(),
        kpoints_cartesian_2pi_over_alat=kpoints.tolist(),
        weights=(weights / weights.sum()).tolist(),
        sampled_gap_eV=gap,
        scf_error_hartree=float(output.findtext("convergence_info/scf_conv/scf_error")),
        total_energy_hartree=float(output.findtext("total_energy/etot")),
    )


def run(root, index, ranks):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if digest(__file__) != m["source_sha256"]:
        raise RuntimeError("Runner source changed after preparation")
    case = m["pilot"] if index == "pilot" else m["cases"][int(index)]
    out = root / "cases" / case["name"]
    exe = root / "qe_source" / "bin" / "pw.x"
    expected_binary = (root / "pw.sha256").read_text().split()[0]
    if digest(exe) != expected_binary:
        raise RuntimeError("QE executable checksum mismatch")
    for symbol, meta in m["pseudopotentials"].items():
        if digest(root / "pseudos" / f"{symbol}.upf") != meta["sha256"]:
            raise RuntimeError("Pseudopotential checksum mismatch")
    start = time.monotonic()
    result = dict(
        case=case["name"],
        job_id=os.environ.get("SLURM_JOB_ID"),
        ranks=ranks,
        executable_sha256=expected_binary,
        source_sha256=digest(__file__),
        manifest_sha256=digest(root / "manifest.json"),
        functionals={},
    )
    try:
        for functional in ("PBE", "PBE0"):
            folder = out / functional
            spec = case["inputs"][functional]
            if digest(spec["path"]) != spec["sha256"]:
                raise RuntimeError("QE input changed after preparation")
            if spec["restart_from_pbe"]:
                (folder / "scratch").mkdir()
                shutil.copytree(save_directory(out / "PBE"), save_directory(folder))
            # Four k pools; hybrid exchange also uses four band groups. The
            # subspace eigensolve stays serial: this local build fails Cholesky
            # with -ndiag 4, whereas both band-group layouts pass with -ndiag 1.
            bands = 4 if functional == "PBE0" and ranks >= 32 else 1
            command = [
                "mpirun",
                "-np",
                str(ranks),
                "--bind-to",
                "core",
                str(exe),
                "-nk",
                "4",
                "-nb",
                str(bands),
                "-ndiag",
                "1",
                "-in",
                spec["path"],
            ]
            print(
                json.dumps(
                    {"case": case["name"], "functional": functional, "command": command}
                ),
                flush=True,
            )
            began = time.monotonic()
            with (folder / "qe.log").open("x") as log:
                subprocess.run(
                    command,
                    cwd=folder,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(60, 6900 - (time.monotonic() - start)),
                )
            xml = save_directory(folder) / "data-file-schema.xml"
            report = read_spectrum(
                xml, folder / "qe.log", hybrid=functional == "PBE0", nocc=case["nocc"]
            )
            if len(report["eigenvalues_eV"]) != int(np.prod(case["kmesh"])):
                raise RuntimeError("Unexpected k-point count")
            report.update(
                wall_seconds=time.monotonic() - began,
                xml_sha256=digest(xml),
                mpi_command=command,
            )
            save_json(folder / "report.json", report)
            result["functionals"][functional] = report
        if index == "pilot":
            comparisons = {}
            for functional, report in result["functionals"].items():
                with np.load(
                    Path(case["legacy_sample"]) / functional / "exchange.npz",
                    allow_pickle=False,
                ) as old:
                    error = float(
                        np.max(
                            np.abs(
                                np.array(report["eigenvalues_eV"]) - old["eigenvalues"]
                            )
                        )
                    )
                comparisons[functional] = dict(
                    max_eigenvalue_difference_eV=error, tolerance_eV=1e-3
                )
            result["legacy_comparison"] = comparisons
            if any(
                v["max_eigenvalue_difference_eV"] > v["tolerance_eV"]
                for v in comparisons.values()
            ):
                raise RuntimeError("Cross-version eigenvalue comparison exceeds 1 meV")
        result.update(complete=True, wall_seconds=time.monotonic() - start)
        save_json(out / "completion.json", result)
        print(
            json.dumps(
                {"complete": case["name"], "wall_seconds": result["wall_seconds"]}
            ),
            flush=True,
        )
    except Exception as exc:
        result.update(
            complete=False,
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.monotonic() - start,
        )
        save_json(out / "failure.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--pseudo-dir", required=True)
    p.add_argument("--legacy-sample", required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.root, args.pseudo_dir, args.legacy_sample)
        print(f"Prepared {len(m['cases'])} supercells and one primitive validation")
    else:
        run(args.root, args.index, args.ranks)


if __name__ == "__main__":
    main()
