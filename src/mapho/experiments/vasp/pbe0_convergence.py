"""Si PBE0 mesh checks at common k points before conditional PAW-head training."""

import itertools
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from ase.build import bulk
import h5py
import numpy as np

from ...source_snapshot import snapshot_package, source_path
from ...vasp.ace import export_hybrid, prepare_hybrid_replay
from ...vasp.paw import (
    compare,
    prepare_reference,
    read_bands,
    run,
    save_json,
    sha,
    write_incar,
)
from .hybrid_pilot import LAUNCHER, INPUTS

THRESHOLDS = dict(
    energy_per_atom_eV=0.001,
    aligned_frontier_max_eV=0.005,
    potential_rmse_eV=0.005,
    onsite_rmse_eV=0.005,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def calibration_structure(label):
    atoms = bulk("Si", "diamond", a=5.43)
    if label == "compressed":
        atoms.set_cell(atoms.cell.array * 0.985, scale_atoms=True)
    elif label == "expanded_displaced":
        atoms.set_cell(atoms.cell.array * 1.015, scale_atoms=True)
        atoms.positions += np.array([[0.01, -0.01, 0.0], [-0.01, 0.01, 0.0]])
    elif label != "equilibrium":
        raise ValueError("Unknown calibration structure")
    atoms.wrap()
    return atoms


def prepare(root, potcar, executable, meshes=(4, 6, 8)):
    meshes = list(meshes)
    if (
        len(meshes) < 3
        or meshes != sorted(set(meshes))
        or any(int(n) != n or n < 2 or n % 2 for n in meshes)
    ):
        raise ValueError("Supply at least three increasing even meshes")
    potcar, executable = Path(potcar).resolve(), Path(executable).resolve()
    if not potcar.is_file() or not executable.is_file():
        raise FileNotFoundError("PAW data and the validated executable are required")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    cases = []
    for label in ("equilibrium", "compressed", "expanded_displaced"):
        for mesh in meshes:
            relative = f"{label}/mesh_{mesh}"
            case = root / relative
            for functional in ("PBE", "PBE0"):
                folder = case / functional.lower()
                m = prepare_reference(
                    folder,
                    calibration_structure(label),
                    {"Si": potcar},
                    mesh=(mesh,) * 3,
                    encut=600.0,
                    nbands=12,
                    algorithm="Normal",
                )
                settings = m["settings"]
                settings.update(EDIFF=1e-10, NELM=240, NELMIN=12)
                if functional == "PBE0":
                    settings.update(
                        SYSTEM="MAPHO Si PBE0 mesh convergence",
                        LHFCALC=True,
                        AEXX=0.25,
                        HFSCREEN=0.0,
                        HFRCUT=0.0,
                        PRECFOCK="Accurate",
                        ALGO="Normal",
                        LFOCKACE=True,
                        IMIX=1,
                        AMIX=0.2,
                        NELMIN=24,
                        ISTART=1,
                        ICHARG=0,
                        NELMDL=0,
                    )
                    (folder / "MAPHO_FOCK_EXPORT").touch()
                write_incar(folder / "INCAR", settings)
                m.update(
                    functional=functional,
                    purpose="Mesh convergence before conditional PBE0 learning",
                )
                m["inputs"] = {name: sha(folder / name) for name in INPUTS}
                if functional == "PBE0":
                    m["inputs"]["MAPHO_FOCK_EXPORT"] = sha(folder / "MAPHO_FOCK_EXPORT")
                save_json(folder / "manifest.json", m)
            cases.append(dict(directory=relative, structure=label, mesh=mesh))
    hashes = snapshot_package(root / "source_snapshot/mapho")
    m = dict(
        schema="mapho.si_pbe0_convergence.v1",
        cases=cases,
        meshes=meshes,
        executable=str(executable),
        executable_sha256=sha(executable),
        potcar=str(potcar),
        potcar_sha256=sha(potcar),
        source_sha256=hashes,
        thresholds=THRESHOLDS,
        replay_tolerance_eV=1e-4,
        exchange=dict(fraction=0.25, HFSCREEN=0.0, HFRCUT=0.0, reduced_mesh=False),
        training_scope="Predict smooth/PAW components; reference ACE exchange is retained",
        selection_rule="Use the finest mesh only when its comparison with the next-finest passes on all three geometries",
        sampled_gap_is_convergence_metric=False,
    )
    save_json(root / "manifest.json", m)
    return m


def verify_source(root):
    m = read_json(Path(root) / "manifest.json")
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError(f"Worker source changed: {name}")
    if sha(m["executable"]) != m["executable_sha256"]:
        raise ValueError("Validated executable changed")
    return m


def worker(root, index, timeout=162000):
    root = Path(root).resolve()
    m = verify_source(root)
    if not 0 <= index < len(m["cases"]):
        raise ValueError("Calibration case index is out of range")
    spec = m["cases"][index]
    case = root / spec["directory"]
    report = dict(
        complete=False, passed=False, job=os.environ.get("SLURM_JOB_ID"), **spec
    )
    save_json(case / "status.json", report)

    def calculate(folder):
        result = run(folder, m["executable"], launcher=LAUNCHER, timeout=timeout)
        if "Offloading initialized" not in (folder / "stdout.log").read_text():
            raise RuntimeError("CUDA offload not reported")
        return result

    try:
        report["pbe"] = calculate(case / "pbe")
        ref = case / "pbe0"
        shutil.copyfile(case / "pbe/WAVECAR", ref / "WAVECAR")
        rm = read_json(ref / "manifest.json")
        rm["inputs"]["WAVECAR"] = sha(ref / "WAVECAR")
        rm["restart_source"] = str(case / "pbe/WAVECAR")
        save_json(ref / "manifest.json", rm)
        report["pbe0"] = calculate(ref)
        rm["initial_wavecar_sha256"] = rm["inputs"].pop("WAVECAR")
        rm["final_wavecar_sha256"] = sha(ref / "WAVECAR")
        save_json(ref / "manifest.json", rm)
        export_hybrid(ref, case / "operator.h5")
        prepare_hybrid_replay(case / "operator.h5", case / "replay")
        report["replay_run"] = calculate(case / "replay")
        report["replay"] = compare(
            ref, case / "replay", tolerance_eV=m["replay_tolerance_eV"]
        )
        if not report["replay"]["passed"]:
            raise RuntimeError("Frozen reference exchange replay failed")
        report.update(complete=True, passed=True)
        report["output_sha256"] = {
            str(p.relative_to(case)): sha(p)
            for p in (
                case / "operator.h5",
                case / "operator_components.h5",
                case / "operator.ace.bin",
                ref / "EIGENVAL",
                ref / "vasprun.xml",
                case / "replay/EIGENVAL",
            )
        }
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save_json(case / "status.json", report)
    return report


def common_frontier(bands):
    """Use the same eight reciprocal half-grid points on every even mesh."""
    ids = []
    for point in itertools.product((0.0, 0.5), repeat=3):
        delta = bands["kpoints"] - point
        delta -= np.rint(delta)
        matches = np.flatnonzero(np.max(np.abs(delta), axis=1) < 1e-7)
        if len(matches) != 1:
            raise ValueError("Expected exactly one copy of each common k point")
        ids.append(matches[0])
    levels = bands["eigenvalues_eV"][ids]
    if levels.ndim != 2 or levels.shape[1] < 6:
        raise ValueError("Expected four occupied and at least two empty Si bands")
    return levels[:, 2:6] - levels[:, 3].max()


def compare_meshes(a, b, thresholds=THRESHOLDS):
    a, b = Path(a), Path(b)
    for case in (a, b):
        status = read_json(case / "status.json")
        if not status["complete"] or not status["passed"]:
            raise ValueError("Incomplete native reference/replay")
        for name, digest in status["output_sha256"].items():
            if sha(case / name) != digest:
                raise ValueError("Native convergence result changed")
    levels = [common_frontier(read_bands(case / "pbe0")) for case in (a, b)]
    energy = []
    for case in (a, b):
        xml = ET.parse(case / "pbe0/vasprun.xml")
        energy.append(
            float(xml.findall('./calculation/energy/i[@name="e_0_energy"]')[-1].text)
        )
    with (
        h5py.File(a / "operator_components.h5") as fa,
        h5py.File(b / "operator_components.h5") as fb,
    ):
        for key in ("cell_A", "positions_A", "atomic_numbers", "paw/q", "paw/types"):
            if not np.allclose(fa[key][:], fb[key][:], rtol=0, atol=1e-10):
                raise ValueError(
                    "Convergence comparison requires identical geometry and PAW data"
                )
        va, vb = fa["potential_eV"][:], fb["potential_eV"][:]
        if va.shape != vb.shape:
            raise ValueError("Convergence comparison requires the same native FFT grid")
        shift = float((va - vb).mean())
        dv = va - vb - shift
        dd = fa["paw/d"][:] - fb["paw/d"][:] - shift * fa["paw/q"][:]
        result = dict(
            energy_per_atom_eV=abs(energy[0] - energy[1]) / len(fa["atomic_numbers"]),
            aligned_frontier_max_eV=float(np.abs(levels[0] - levels[1]).max()),
            potential_rmse_eV=float(np.mean(dv**2) ** 0.5),
            onsite_rmse_eV=float(np.mean(dd**2) ** 0.5),
            potential_gauge_shift_eV=shift,
        )
    result["passed"] = all(result[k] <= limit for k, limit in thresholds.items())
    return result


def collect(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    pending = [
        case["directory"]
        for case in m["cases"]
        if not (root / case["directory"] / "status.json").exists()
        or not read_json(root / case["directory"] / "status.json").get("complete")
    ]
    result = dict(
        complete=not pending,
        passed=False,
        pending=pending,
        selected_mesh=None,
        comparisons={},
    )
    if not pending:
        meshes = m["meshes"]
        labels = sorted({case["structure"] for case in m["cases"]})
        for lo, hi in itertools.combinations(meshes, 2):
            key = f"{lo}_vs_{hi}"
            result["comparisons"][key] = {
                label: compare_meshes(
                    root / label / f"mesh_{lo}",
                    root / label / f"mesh_{hi}",
                    m["thresholds"],
                )
                for label in labels
            }
        lo, hi = meshes[-2:]
        if all(
            result["comparisons"][f"{lo}_vs_{hi}"][label]["passed"] for label in labels
        ):
            result.update(passed=True, selected_mesh=hi)
        if not result["passed"]:
            result["next_action"] = "Add denser meshes; do not start training"
    save_json(root / "convergence.json", result)
    return result
