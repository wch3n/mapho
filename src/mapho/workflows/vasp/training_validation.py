"""Native convention checks and held-out PAW operator/eigenvalue validation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
from ase.io import read

from ...vasp.paw import (
    prepare_reference,
    prepare_replay,
    run,
    export_operator,
    read_bands,
    sha,
    save_json,
    write_incar,
)

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
KS = np.array([[0.0, 0.0, 0.0], [0.13, 0.21, 0.07], [0.31, 0.11, 0.23]])


def native_checks(root):
    from ...vasp.model import native_sp_order

    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    entry = m["entries"]["train"][0]
    baseline = Path(entry["baseline"])
    source = baseline.parent
    dataset = json.loads((Path(m["dataset"]) / "manifest.json").read_text())
    executable = dataset["executable"]
    atoms = read(source / "POSCAR")
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.47
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    with h5py.File(baseline) as f:
        v = f["potential_eV"][:]
        d = f["paw/d"][:]
        q = f["paw/q"][:]
    checks = {}
    for name in ("rotation", "k_independence"):
        transformed = atoms.copy()
        if name == "rotation":
            transformed.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
            transformed.positions = atoms.positions @ rotation.T
        directory = root / "native_checks" / name
        manifest = prepare_reference(
            directory,
            transformed,
            {"Si": source / "POTCAR"},
            mesh=(1, 1, 1),
            encut=600.0,
            nbands=48,
        )
        settings = manifest["settings"]
        settings.update(
            ICHARG=12,
            NELMDL=0,
            NELMIN=16,
            LWAVE=False,
            LCHARG=False,
            NGXF=v.shape[0],
            NGYF=v.shape[1],
            NGZF=v.shape[2],
        )
        write_incar(directory / "INCAR", settings)
        if name == "k_independence":
            (directory / "KPOINTS").write_text(
                "atomic-density k independence\n2\nReciprocal\n0 0 0 1\n.13 .21 .07 1\n"
            )
        manifest["inputs"] = {
            f: sha(directory / f) for f in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
        }
        save_json(directory / "manifest.json", manifest)
        run(directory, executable, launcher=LAUNCHER, timeout=1200)
        export_operator(directory, directory / "operator.h5")
        matrix = np.eye(8)
        if name == "rotation":
            matrix[2:5, 2:5] = rotation
            matrix[5:8, 5:8] = rotation
            order = native_sp_order([0, 0, 1, 1])
            matrix = matrix[order][:, order]
        with h5py.File(directory / "operator.h5") as f:
            ve = float(np.abs(f["potential_eV"][:] - v).max())
            de = float(np.abs(f["paw/d"][:] - matrix @ d @ matrix.T).max())
            qe = float(np.abs(f["paw/q"][:] - matrix @ q @ matrix.T).max())
        checks[name] = dict(
            potential_max_eV=ve,
            onsite_max_eV=de,
            overlap_max=qe,
            passed=bool(ve < 1e-5 and de < 1e-5 and qe < 1e-10),
        )
    result = dict(checks=checks, passed=all(x["passed"] for x in checks.values()))
    save_json(root / "native_checks.json", result)
    if not result["passed"]:
        raise RuntimeError("Native convention check failed")
    return result


def bands_case(root, index):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    entry = m["entries"]["test"][index]
    dataset = json.loads((Path(m["dataset"]) / "manifest.json").read_text())
    executable = dataset["executable"]
    folder = root / "bands" / entry["name"]
    folder.mkdir(parents=True, exist_ok=False)
    target = Path(entry["target"])
    baseline = Path(entry["baseline"])
    spectra = {}
    for variant in ("reference", "atomic", "local", "none", "full"):
        kwargs = {}
        operator = target if variant == "reference" else baseline
        if variant not in ("reference", "atomic"):
            with np.load(root / variant / "predictions" / f"{entry['name']}.npz") as f:
                kwargs = dict(potential=f["potential"], d=f["d"])
        prepare_replay(operator, folder / variant, kpoints=KS, **kwargs)
        run(folder / variant, executable, launcher=LAUNCHER, timeout=1200)
        spectra[variant] = read_bands(folder / variant)["eigenvalues_eV"]
    # Independent native frozen-density solve checks the exported reference
    # at the same three k points, including two absent from the training mesh.
    native = folder / "native"
    native.mkdir()
    for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS"):
        shutil.copyfile(folder / "reference" / name, native / name)
    shutil.copyfile(target.parent / "CHGCAR", native / "CHGCAR")
    settings = json.loads((folder / "reference" / "manifest.json").read_text())[
        "settings"
    ]
    settings["ICHARG"] = 11
    write_incar(native / "INCAR", settings)
    save_json(
        native / "manifest.json",
        dict(
            unpatched_control=True,
            settings=settings,
            inputs={
                name: sha(native / name)
                for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
            },
        ),
    )
    run(native, executable, launcher=LAUNCHER, timeout=1200)
    spectra["native"] = read_bands(native)["eigenvalues_eV"]
    nocc = 32
    window = slice(nocc - 4, nocc + 4)
    rows = {}
    for variant, e in spectra.items():
        delta = e - spectra["native"]
        gap = e[:, nocc].min() - e[:, nocc - 1].max()
        refgap = spectra["native"][:, nocc].min() - spectra["native"][:, nocc - 1].max()
        rows[variant] = dict(
            frontier_mae_eV=float(np.abs(delta[:, window]).mean()),
            frontier_max_eV=float(np.abs(delta[:, window]).max()),
            all_band_mae_eV=float(np.abs(delta).mean()),
            sampled_gap_error_eV=float(gap - refgap),
        )
    result = dict(
        name=entry["name"],
        kpoints=KS.tolist(),
        metrics=rows,
        reference_frontier_replay_passed=rows["reference"]["frontier_max_eV"] < 0.001,
        eigenvalues_eV={k: v.tolist() for k, v in spectra.items()},
    )
    save_json(folder / "report.json", result)
    if not result["reference_frontier_replay_passed"]:
        raise RuntimeError(
            "Native/reference replay disagreement exceeds 1 meV in frontier window"
        )
    return result


def parallel(root, commands, devices):
    root = Path(root)
    waiting = list(commands)
    running = {}
    failed = []
    while waiting or running:
        for device in devices:
            if device in running or not waiting:
                continue
            label, cmd = waiting.pop(0)
            log = (root / "logs" / f"{label}.log").open("w")
            process = subprocess.Popen(
                cmd,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[device] = (process, log, label)
            print("Started", label, "GPU", device, flush=True)
        for device, (process, log, label) in list(running.items()):
            if process.poll() is not None:
                log.close()
                del running[device]
                print("Finished", label, "exit", process.returncode, flush=True)
                if process.returncode:
                    failed.append(label)
        if running:
            time.sleep(3)
    if failed:
        raise RuntimeError(f"Workers failed: {failed}")


def pipeline(root, epochs=300):
    root = Path(root).resolve()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mapho.commands.vasp.training_validation",
            "native-checks",
            str(root),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]),
        check=True,
    )
    commands = [
        (
            f"train_{mode}",
            [
                sys.executable,
                "-u",
                "-m",
                "mapho.commands.vasp.training_validation",
                "fit",
                str(root),
                mode,
                "--epochs",
                str(epochs),
            ],
        )
        for mode in ("local", "none", "full")
    ]
    parallel(root, commands, devices)
    m = json.loads((root / "manifest.json").read_text())
    parallel(
        root,
        [
            (
                f"bands_{i}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.commands.vasp.training_validation",
                    "bands",
                    str(root),
                    str(i),
                ],
            )
            for i in range(len(m["entries"]["test"]))
        ],
        devices,
    )
    rows = [
        json.loads((root / "bands" / e["name"] / "report.json").read_text())
        for e in m["entries"]["test"]
    ]
    result = dict(
        complete=True,
        job=os.environ.get("SLURM_JOB_ID"),
        test_structures=len(rows),
        native_checks=json.loads((root / "native_checks.json").read_text()),
        fields={
            mode: json.loads((root / mode / "test_fields.json").read_text())
            for mode in ("local", "none", "full")
        },
        bands={
            mode: {
                metric: float(np.mean([r["metrics"][mode][metric] for r in rows]))
                for metric in ("frontier_mae_eV", "all_band_mae_eV")
            }
            for mode in ("reference", "atomic", "local", "none", "full")
        },
    )
    save_json(root / "completion.json", result)
