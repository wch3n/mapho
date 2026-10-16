"""Run a bounded subset of an already frozen reference dataset on debug GPUs."""

import json
import os
from pathlib import Path
import sys

from .material_dataset import convergence, audit
from .training_validation import parallel
from ...vasp.paw import sha, save_json


def execute(root, start, stop):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if not 0 <= start < stop <= len(m["samples"]):
        raise ValueError("Invalid sample interval")
    if not convergence(root)["passed"]:
        raise ValueError("K-point gate failed")
    commands = []
    for i in range(start, stop):
        folder = Path(m["samples"][i]["directory"])
        if (folder / "report.json").exists():
            r = json.loads((folder / "report.json").read_text())
            if r["complete"] and sha(folder / "operator.h5") == r["operator_sha256"]:
                continue
        if (folder / "run.json").exists():
            raise RuntimeError(f"Uncertified previous run: {folder}")
        commands.append(
            (
                f"samples_{i:03d}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.commands.vasp.data",
                    "material",
                    "worker",
                    str(root),
                    "samples",
                    str(i),
                ],
            )
        )
    parallel(root, commands, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    save_json(
        root / f"chunk_{start:03d}_{stop:03d}.json",
        dict(complete=True, start=start, stop=stop, job=os.environ.get("SLURM_JOB_ID")),
    )
    if all((Path(row["directory"]) / "report.json").exists() for row in m["samples"]):
        audit(root)
