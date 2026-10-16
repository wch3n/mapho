"""Native PBE0 component diagnostic before extending the learned PAW operator.

This is not an ML fit: exact native smooth/on-site components are replayed
with Fock omitted to measure the missing contribution of the current interface.
All executable, input, source and output identities are retained in scratch.
"""

from ...source_snapshot import snapshot_package, source_hashes, source_path
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from ase.build import bulk
import numpy as np

from ...vasp.paw import (
    compare,
    export_operator,
    prepare_reference,
    prepare_replay,
    read_bands,
    run,
    save_json,
    sha,
    write_incar,
)


INPUTS = ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
LAUNCHER = [
    "mpirun",
    "--bind-to",
    "none",
    "--mca",
    "pml",
    "ob1",
    "--mca",
    "btl",
    "self,vader",
    "-np",
    "1",
]


def prepare(root, potcar, executable, native_executable, meshes=(2, 3)):
    root = Path(root).resolve()
    if (
        not meshes
        or any(int(n) != n or n < 2 for n in meshes)
        or len(set(meshes)) != len(meshes)
    ):
        raise ValueError("Provide distinct integer meshes of at least two")
    for path in (potcar, executable, native_executable):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot" / "mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    cases = []
    for n in meshes:
        case = root / f"mesh_{n}"
        for functional in ("PBE", "PBE0"):
            folder = case / functional.lower()
            m = prepare_reference(
                folder,
                bulk("Si", "diamond", a=5.43),
                {"Si": potcar},
                encut=600.0,
                mesh=(n, n, n),
                nbands=12,
                algorithm="Exact",
                ispin=1,
            )
            settings = m["settings"]
            settings.update(NELMIN=12, NELM=240, EDIFF=1e-9)
            if functional == "PBE0":
                settings.update(
                    SYSTEM="MAPHO Si PBE0 component diagnostic",
                    LHFCALC=True,
                    AEXX=0.25,
                    HFSCREEN=0.0,
                    HFRCUT=0.0,
                    PRECFOCK="Accurate",
                    ALGO="Damped",
                    TIME=0.4,
                    LFOCKACE=False,
                    ISTART=1,
                    ICHARG=0,
                    NELMDL=0,
                )
            write_incar(folder / "INCAR", settings)
            m.update(
                functional=functional,
                purpose="Native component diagnostic; not converged training data",
                inputs={name: sha(folder / name) for name in INPUTS},
            )
            save_json(folder / "manifest.json", m)
        cases.append(dict(name=case.name, mesh=[n] * 3))
    manifest = dict(
        schema="mapho.vasp_pbe0_component_pilot.v1",
        cases=cases,
        material="Si",
        natoms=2,
        lattice_constant_A=5.43,
        encut_eV=600.0,
        executable=str(Path(executable).resolve()),
        executable_sha256=sha(executable),
        native_executable=str(Path(native_executable).resolve()),
        native_executable_sha256=sha(native_executable),
        potcar_sha256=sha(potcar),
        source_sha256=source_hashes(snapshot),
        tolerance_eV=1e-4,
        training_started=False,
        purpose="Test what the present smooth-potential/PAW representation omits for PBE0",
        exchange=dict(
            fraction=0.25,
            screening_inv_A=0.0,
            real_space_cutoff_A=0.0,
            reduced_exchange_mesh=False,
            ace=False,
            solver="Damped, TIME=0.4; native full exchange",
        ),
        limitations=[
            "Small meshes are diagnostic, not a convergence certificate.",
            "No learned PBE0 Hamiltonian or speedup is claimed.",
            "Omitting Fock measures a missing term, not a lower bound for a fitted effective Hamiltonian.",
            "PBE PAW datasets are retained for the PBE0 references.",
        ],
        references=[
            "https://vasp.at/wiki/LHFCALC",
            "https://vasp.at/wiki/Band-structure_calculation_using_hybrid_functionals",
        ],
    )
    save_json(root / "manifest.json", manifest)
    return manifest


def band_metrics(reference, candidate, nocc=4):
    a, b = read_bands(reference), read_bands(candidate)
    x, y = a["eigenvalues_eV"], b["eigenvalues_eV"]
    if x.shape != y.shape or not np.allclose(
        a["kpoints"], b["kpoints"], rtol=0, atol=1e-10
    ):
        raise ValueError("Band diagnostic requires identical k points and band counts")
    if x.ndim != 2 or not 0 < nocc < x.shape[1]:
        raise ValueError("Expected nonmagnetic occupied and empty bands")
    vbm_x, vbm_y = x[:, nocc - 1].max(), y[:, nocc - 1].max()
    gap_x = x[:, nocc].min() - vbm_x
    gap_y = y[:, nocc].min() - vbm_y
    error = y - x
    aligned = error - (vbm_y - vbm_x)
    window = slice(max(0, nocc - 2), min(x.shape[1], nocc + 2))
    return dict(
        all_band_mae_eV=float(np.abs(error).mean()),
        frontier_mae_eV=float(np.abs(error[:, window]).mean()),
        vbm_aligned_frontier_mae_eV=float(np.abs(aligned[:, window]).mean()),
        reference_sampled_gap_eV=float(gap_x),
        candidate_sampled_gap_eV=float(gap_y),
        sampled_gap_error_eV=float(gap_y - gap_x),
        vbm_shift_eV=float(vbm_y - vbm_x),
        frontier_band_indices=list(range(x.shape[1]))[window],
    )


def execute(root, index):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    for key in ("executable", "native_executable"):
        if sha(m[key]) != m[key + "_sha256"]:
            raise ValueError(f"{key} changed")
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError(f"Worker source changed: {name}")
    case = root / m["cases"][index]["name"]
    report = dict(
        complete=False,
        training_started=False,
        job=os.environ.get("SLURM_JOB_ID"),
        array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        mesh=m["cases"][index]["mesh"],
    )
    save_json(case / "status.json", report)

    def calculate(folder, native=False):
        executable = m["native_executable" if native else "executable"]
        result = run(folder, executable, launcher=LAUNCHER, timeout=3300)
        if "Offloading initialized" not in (folder / "stdout.log").read_text():
            raise RuntimeError("Native CUDA offload was not reported")
        return result

    try:
        calculate(case / "pbe")
        export_operator(case / "pbe", case / "pbe_operator.h5")
        prepare_replay(case / "pbe_operator.h5", case / "pbe_replay")
        calculate(case / "pbe_replay")
        report["pbe_replay"] = compare(
            case / "pbe", case / "pbe_replay", tolerance_eV=m["tolerance_eV"]
        )
        if not report["pbe_replay"]["passed"]:
            raise RuntimeError("PBE replay control failed")
        # Same-mesh PBE orbitals initialize the full hybrid SCF; density is updated.
        pbe0 = case / "pbe0"
        shutil.copyfile(case / "pbe" / "WAVECAR", pbe0 / "WAVECAR")
        pm = json.loads((pbe0 / "manifest.json").read_text())
        pm["inputs"]["WAVECAR"] = sha(pbe0 / "WAVECAR")
        pm["restart_source"] = str(case / "pbe" / "WAVECAR")
        save_json(pbe0 / "manifest.json", pm)
        calculate(pbe0)
        # WAVECAR is a read/write restart file: preserve the initial input hash
        # in run provenance, while the exported manifest tracks static inputs.
        pm["initial_wavecar_sha256"] = pm["inputs"].pop("WAVECAR")
        pm["final_wavecar_sha256"] = sha(pbe0 / "WAVECAR")
        save_json(pbe0 / "manifest.json", pm)
        export_operator(pbe0, case / "pbe0_components.h5", components_only=True)

        native = case / "pbe0_native_restart"
        native.mkdir()
        for name in (*INPUTS, "WAVECAR"):
            shutil.copyfile(pbe0 / name, native / name)
        nm = dict(pm, unpatched_control=True)
        nm["purpose"] = (
            "Unmodified VASP continuation: bridge invariance and SCF restart stability"
        )
        nm["inputs"] = {name: sha(native / name) for name in (*INPUTS, "WAVECAR")}
        save_json(native / "manifest.json", nm)
        calculate(native, native=True)
        report["pbe0_native_restart"] = compare(
            pbe0, native, tolerance_eV=m["tolerance_eV"]
        )
        if not report["pbe0_native_restart"]["passed"]:
            raise RuntimeError("PBE0 bridge/restart stability check failed")

        omitted = case / "pbe0_without_fock"
        om = prepare_replay(case / "pbe0_components.h5", omitted, omit_fock=True)
        report["d_hermitization_max_eV"] = om["d_hermitization_max_eV"]
        calculate(omitted)
        report["pbe_vs_pbe0"] = band_metrics(pbe0, case / "pbe")
        report["exact_components_without_fock_vs_pbe0"] = band_metrics(pbe0, omitted)
        with (case / "bands.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["calculation", "k_index", "kx", "ky", "kz", "band_index", "energy_eV"]
            )
            for label in ("pbe", "pbe0", "pbe0_native_restart", "pbe0_without_fock"):
                bands = read_bands(case / label)
                for k, point in enumerate(bands["kpoints"]):
                    for band, energy in enumerate(bands["eigenvalues_eV"][k]):
                        writer.writerow([label, k, *point, band, energy])
        report.update(
            complete=True,
            interface_controls_passed=True,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={
                str(p.relative_to(case)): sha(p)
                for p in case.rglob("*")
                if p.is_file()
                and p.name
                in (
                    "EIGENVAL",
                    "POT",
                    "MLHAM_PAW.dat",
                    "bands.csv",
                    "pbe0_components.h5",
                    "pbe_operator.h5",
                )
            },
        )
    except Exception as exc:
        report.update(error=repr(exc), interface_controls_passed=False)
        raise
    finally:
        save_json(case / "status.json", report)
    return report


def collect(root):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    reports = {
        c["name"]: json.loads((root / c["name"] / "status.json").read_text())
        for c in m["cases"]
    }
    if not all(
        r["complete"] and r["interface_controls_passed"] for r in reports.values()
    ):
        raise ValueError(
            "Every native case must finish and pass its interface controls"
        )
    report = dict(
        complete=True,
        training_started=False,
        cases=reports,
        kpoint_convergence_certified=False,
        learned_pbe0_hamiltonian=False,
        interpretation="Exact native smooth/PAW components were tested with Fock deliberately omitted. "
        "This does not measure ML prediction error or the best effective fit within this architecture.",
    )
    save_json(root / "completion.json", report)
    return report
