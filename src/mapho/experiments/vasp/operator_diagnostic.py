"""Separate learned smooth-potential and PAW-matrix errors in frozen operators."""

from ...source_snapshot import snapshot_package, source_hashes, source_path
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

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
from .training_validation import LAUNCHER, parallel


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(root, training):
    root, training = Path(root).resolve(), Path(training).resolve()
    m = read_json(training / "silu/manifest.json")
    dataset = read_json(Path(m["dataset"]) / "manifest.json")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    entries = [
        e for e in m["entries"]["test"] if e["name"] in ("sample_020", "sample_023")
    ]
    tasks = []
    for e in entries:
        for variant in (
            "reference_tight",
            "v_only",
            "d_only",
            "full_tight",
            "native_export",
        ):
            tasks.append(dict(name=e["name"], variant=variant))
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    manifest = dict(
        training=str(training),
        entries=entries,
        tasks=tasks,
        executable=dataset["executable"],
        executable_sha256=dataset["executable_sha256"],
        nocc=int(dataset["nelect"] / 2),
        predictions={
            e["name"]: dict(
                path=str(training / "silu/full/predictions" / f"{e['name']}.npz"),
                sha256=sha(training / "silu/full/predictions" / f"{e['name']}.npz"),
            )
            for e in entries
        },
        source_sha256=source_hashes(snapshot),
        purpose="Diagnosis only; no training or threshold changes; original results retained",
    )
    save_json(root / "manifest.json", manifest)


def worker(root, index):
    root = Path(root)
    m = read_json(root / "manifest.json")
    task = m["tasks"][index]
    entry = next(e for e in m["entries"] if e["name"] == task["name"])
    target = Path(entry["target"])
    pred = m["predictions"][entry["name"]]
    variant = task["variant"]
    folder = root / entry["name"] / variant
    for path, digest in [
        (target, entry["target_sha256"]),
        (pred["path"], pred["sha256"]),
        (m["executable"], m["executable_sha256"]),
    ]:
        if sha(path) != digest:
            raise ValueError("Diagnostic source changed")
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError("Diagnostic snapshot changed")
    old = read_json(Path(m["training"]) / "bands" / entry["name"] / "report.json")
    points = np.array(old["kpoints"])
    if variant == "native_export":
        from ase.io import read

        with h5py.File(target) as f:
            original = json.loads(f.attrs["metadata_json"])["reference_manifest"]
            shape = f["potential_eV"].shape
        potcars = {p["symbol"]: p["path"] for p in original["potentials"]}
        manifest = prepare_reference(
            folder,
            read(target.parent / "POSCAR"),
            potcars,
            encut=original["settings"]["ENCUT"],
            nbands=original["settings"]["NBANDS"],
            mesh=(1, 1, 1),
        )
        shutil.copyfile(target.parent / "CHGCAR", folder / "CHGCAR")
        shutil.copyfile(
            Path(m["training"]) / "bands" / entry["name"] / "native/KPOINTS",
            folder / "KPOINTS",
        )
        settings = manifest["settings"]
        settings.update(
            ICHARG=11,
            NELMDL=0,
            NELMIN=24,
            EDIFF=1e-11,
            LWAVE=False,
            LCHARG=False,
            NGXF=shape[0],
            NGYF=shape[1],
            NGZF=shape[2],
        )
        write_incar(folder / "INCAR", settings)
        manifest["inputs"] = {
            n: sha(folder / n)
            for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
        }
        save_json(folder / "manifest.json", manifest)
        run(folder, m["executable"], launcher=LAUNCHER, timeout=1500)
        export_operator(folder, folder / "operator.h5")
        replay = folder / "replay"
        prepare_replay(folder / "operator.h5", replay, kpoints=points)
        run(replay, m["executable"], launcher=LAUNCHER, timeout=1500)
        a = read_bands(folder)["eigenvalues_eV"]
        b = read_bands(replay)["eigenvalues_eV"]
        with h5py.File(target) as t, h5py.File(folder / "operator.h5") as f:
            dv = f["potential_eV"][:] - t["potential_eV"][:]
            dd = f["paw/d"][:] - t["paw/d"][:]
            report = dict(
                variant=variant,
                native_export_replay_max_eV=float(abs(a - b).max()),
                reconstructed_v_rmse_eV=float(np.mean(dv * dv) ** 0.5),
                reconstructed_v_max_eV=float(abs(dv).max()),
                reconstructed_d_max_eV=float(abs(dd).max()),
                eigenvalues_eV=a.tolist(),
            )
    else:
        with np.load(pred["path"]) as f:
            v, d = f["potential"], f["d"]
        kwargs = {}
        if variant in ("v_only", "full_tight"):
            kwargs["potential"] = v
        if variant in ("d_only", "full_tight"):
            kwargs["d"] = d
        manifest = prepare_replay(target, folder, kpoints=points, **kwargs)
        manifest["settings"].update(EDIFF=1e-11, NELMIN=24)
        write_incar(folder / "INCAR", manifest["settings"])
        manifest["inputs"]["INCAR"] = sha(folder / "INCAR")
        save_json(folder / "manifest.json", manifest)
        run(folder, m["executable"], launcher=LAUNCHER, timeout=1500)
        a = read_bands(folder)["eigenvalues_eV"]
        report = dict(variant=variant, eigenvalues_eV=a.tolist())
    save_json(folder / "report.json", report)


def summarize(root):
    from .activation_ablation import band_metrics

    root = Path(root)
    m = read_json(root / "manifest.json")
    reports = {}
    for entry in m["entries"]:
        name = entry["name"]
        ref = np.array(
            read_json(root / name / "reference_tight/report.json")["eigenvalues_eV"]
        )
        rows = {}
        old = read_json(Path(m["training"]) / "bands" / name / "report.json")
        for variant in (
            "reference_tight",
            "v_only",
            "d_only",
            "full_tight",
            "native_export",
        ):
            row = read_json(root / name / variant / "report.json")
            a = np.array(row.pop("eigenvalues_eV"))
            rows[variant] = dict(
                **row,
                metrics=band_metrics(a, ref, m["nocc"], 4),
                lowest_eigenvalue_eV=float(a.min()),
            )
        rows["existing_full_vs_tight_max_eV"] = float(
            abs(
                np.array(old["eigenvalues_eV"]["silu_full"])
                - np.array(
                    read_json(root / name / "full_tight/report.json")["eigenvalues_eV"]
                )
            ).max()
        )
        reports[name] = rows
    save_json(
        root / "completion.json",
        dict(complete=True, job=os.environ.get("SLURM_JOB_ID"), reports=reports),
    )


def execute(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    tasks = [
        (
            f"{t['name']}_{t['variant']}",
            [
                sys.executable,
                "-u",
                "-m",
                "mapho.commands.vasp.validate",
                "operator",
                "worker",
                str(root),
                str(i),
            ],
        )
        for i, t in enumerate(m["tasks"])
    ]
    parallel(root, tasks, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    summarize(root)
