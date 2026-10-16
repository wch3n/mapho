"""Geometry-only PAW reference from fixed superposed atomic densities.

ICHARG=12 keeps both the initial smooth charge and atomic PAW occupancies
fixed. No target CHGCAR, WAVECAR, potential, D matrix, or eigenvalues is read.
The target FFT dimensions are used only as a common numerical discretization.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
from ase.io import read

from .paw import prepare_reference, run, export_operator, save_json, sha, write_incar


def prepare(dataset, root):
    dataset, root = Path(dataset).resolve(), Path(root).resolve()
    m = json.loads((dataset / "manifest.json").read_text())
    if not json.loads((dataset / "completion.json").read_text())["complete"]:
        raise ValueError("Incomplete reference dataset")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    cases = []
    for sample in m["samples"]:
        source = Path(sample["directory"])
        name = source.name
        atoms = read(source / "POSCAR")
        with h5py.File(source / "operator.h5") as f:
            grid = list(f["potential_eV"].shape)
        folder = root / name
        native = json.loads((source / "manifest.json").read_text())
        potcars = {p["symbol"]: p["path"] for p in native["potentials"]}
        for p in native["potentials"]:
            if sha(p["path"]) != p["sha256"]:
                raise ValueError("PAW dataset changed")
        manifest = prepare_reference(
            folder,
            atoms,
            potcars,
            mesh=(1, 1, 1),
            encut=native["settings"]["ENCUT"],
            nbands=native["settings"]["NBANDS"],
            algorithm="Normal",
        )
        settings = manifest["settings"]
        settings.update(
            ICHARG=12,
            NELMDL=0,
            NELMIN=16,
            LWAVE=False,
            LCHARG=True,
            NGXF=grid[0],
            NGYF=grid[1],
            NGZF=grid[2],
        )
        write_incar(folder / "INCAR", settings)
        manifest.update(
            purpose="geometry-only fixed atomic-density PAW baseline, not an SCF target",
            baseline_definition="VASP ICHARG=12, ISTART=0, no input CHGCAR/WAVECAR",
            inputs={
                n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
            },
        )
        save_json(folder / "manifest.json", manifest)
        cases.append(
            dict(
                name=name,
                folder=str(folder),
                source_geometry_sha256=sha(source / "POSCAR"),
            )
        )
    result = dict(
        schema="mapho.vasp_atomic_reference.v1",
        dataset=str(dataset),
        dataset_manifest_sha256=sha(dataset / "manifest.json"),
        cases=cases,
        executable=m["executable"],
        executable_sha256=m["executable_sha256"],
    )
    save_json(root / "manifest.json", result)
    return result


def worker(root, index):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    folder = Path(m["cases"][index]["folder"])
    if sha(m["executable"]) != m["executable_sha256"]:
        raise ValueError("Executable changed")
    if (folder / "CHGCAR").exists() or (folder / "WAVECAR").exists():
        raise ValueError(
            "Atomic baseline must start without density or wavefunction files"
        )
    run(
        folder,
        m["executable"],
        launcher=[
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
        ],
        timeout=1200,
    )
    export_operator(folder, folder / "operator.h5")
    # Retain a small explicit contract; atomic eigenvalues are never ML inputs.
    result = dict(
        complete=True,
        operator_sha256=sha(folder / "operator.h5"),
        icharg=12,
        target_electronic_files_read=False,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    save_json(folder / "complete.json", result)
    return result


def execute(root):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    queue = list(range(len(m["cases"])))
    running = {}
    failed = []
    while queue or running:
        for device in devices:
            if device in running or not queue:
                continue
            i = queue.pop(0)
            log = (root / "logs" / f"{i:03d}.log").open("w")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.vasp.atomic_reference",
                    "worker",
                    str(root),
                    str(i),
                ],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[device] = (p, log, i)
        for device, (p, log, i) in list(running.items()):
            if p.poll() is not None:
                log.close()
                del running[device]
                print("Atomic reference", i, "exit", p.returncode, flush=True)
                if p.returncode:
                    failed.append(i)
        if running:
            time.sleep(2)
    if failed:
        raise RuntimeError(f"Failed atomic references: {failed}")
    save_json(
        root / "completion.json",
        dict(complete=True, count=len(m["cases"]), job=os.environ.get("SLURM_JOB_ID")),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("dataset")
    q.add_argument("root")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.dataset, a.root)
    elif a.command == "worker":
        worker(a.root, a.index)
    else:
        execute(a.root)
