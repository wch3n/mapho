"""Recover a k/q study without changing live workers or incomplete QE saves."""

from ..source_snapshot import snapshot_package
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from ase.io import read

from .kpoint_convergence import _launch, _reuse, _verify, locked_summary, replace_json
from ..kpoints import mesh_from_spacing
from .supercell_reference import digest, save_json, write_case


def _copy_input(source, destination, old_root, root):
    text = source.read_text().replace(
        str(source.parent / "scratch"), str(destination.parent / "scratch")
    )
    text = text.replace(str(old_root / "pseudos"), str(root / "pseudos"))
    destination.write_text(text)
    return dict(
        path=str(destination),
        sha256=digest(destination),
        restart_from_pbe="/PBE0/" in str(destination),
    )


def prepare(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    old = json.loads((previous / "manifest.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    shutil.copytree(previous / "pseudos", root / "pseudos")
    m = copy.deepcopy(old)
    m.update(
        schema="mapho.kpoint_recovery.v1",
        previous_study=str(previous),
        previous_manifest_sha256=digest(previous / "manifest.json"),
        max_allocations_per_series=1,
        extra_cases=[],
        tasks=[],
        training_authorized=False,
        live_series_left_in_previous_study=[4],
    )
    for series in m["series"]:
        for case in series["cases"]:
            source = previous / "cases" / case["name"]
            destination = root / "cases" / case["name"]
            destination.mkdir(parents=True)
            shutil.copyfile(
                source / "structure.extxyz", destination / "structure.extxyz"
            )
            for functional in ("PBE", "PBE0"):
                old_folder = source / functional
                new_folder = destination / functional
                if (old_folder / "report.json").exists():
                    new_folder.symlink_to(old_folder, target_is_directory=True)
                else:
                    new_folder.mkdir()
                    case["inputs"][functional] = _copy_input(
                        old_folder / "qe.in", new_folder / "qe.in", previous, root
                    )
                    # Only independently completed, verified coarse references
                    # may be imported. Interrupted scratch files are never used.
                    _reuse(root, case, functional)
    # A new transverse refinement on the smallest cells first. These additions
    # do not silently replace the original acceptance criteria in summary.json.
    for series in m["series"][:2]:
        reference = series["cases"][2]
        atoms = read(
            root / "cases" / reference["name"] / "PBE/qe.in", format="espresso-in"
        )
        proposal = mesh_from_spacing(atoms, [0.8, 0.27, 0.27])
        name = series["name"] + "_k1x6x6"
        if proposal["mesh"] != [1, 6, 6]:
            raise RuntimeError("Unexpected new transverse mesh")
        case = write_case(
            root, name, atoms, reference["info"], m["pseudopotentials"], mesh=(1, 6, 6)
        )
        case.update(mesh_proposal=proposal, compare_to=reference["name"])
        m["extra_cases"].append(case)
    # Independent two-node layout validation against a completed 128-rank run.
    reference = m["series"][0]["cases"][0]
    source = root / "cases" / reference["name"]
    validation = copy.deepcopy(reference)
    validation["name"] = "batch_layout_validation"
    validation.pop("reuse_candidate", None)
    destination = root / "cases" / validation["name"]
    destination.mkdir()
    shutil.copyfile(source / "structure.extxyz", destination / "structure.extxyz")
    (destination / "PBE").symlink_to(
        (source / "PBE").resolve(), target_is_directory=True
    )
    (destination / "PBE0").mkdir()
    validation["inputs"]["PBE0"] = _copy_input(
        source / "PBE0/qe.in", destination / "PBE0/qe.in", previous, root
    )
    # source may itself be a symlink with an input pointing at the baseline save.
    text = (destination / "PBE0/qe.in").read_text()
    import re

    text = re.sub(
        r"outdir\s*=\s*'[^']*'",
        lambda _: "outdir = '" + str(destination / "PBE0/scratch") + "'",
        text,
    )
    text = re.sub(
        r"pseudo_dir\s*=\s*'[^']*'",
        lambda _: "pseudo_dir = '" + str(root / "pseudos") + "'",
        text,
    )
    (destination / "PBE0/qe.in").write_text(text)
    validation["inputs"]["PBE0"]["sha256"] = digest(destination / "PBE0/qe.in")
    m["layout_validation"] = validation
    m["layout_reference_report"] = str(source / "PBE0/report.json")
    # The initial batch repairs four interrupted runs, adds the missing 64-atom
    # distorted 3x3 result, and tests 6x6 only on the two smallest geometries.
    initial_batch = set()
    for i, mesh in (
        (0, [2, 4, 4]),
        (1, [2, 4, 4]),
        (2, [1, 3, 3]),
        (3, [1, 3, 3]),
        (5, [1, 3, 3]),
    ):
        initial_batch.add(
            next(c["name"] for c in m["series"][i]["cases"] if c["kmesh"] == mesh)
        )
    initial_batch.update(c["name"] for c in m["extra_cases"])
    for i, series in enumerate(m["series"]):
        if i == 4:
            continue  # The existing worker owns this geometry until it finishes.
        for case in series["cases"]:
            for functional in ("PBE", "PBE0"):
                if not (
                    root / "cases" / case["name"] / functional / "report.json"
                ).exists():
                    m["tasks"].append(
                        dict(
                            case=case["name"],
                            functional=functional,
                            submit_initially=functional == "PBE"
                            or case["name"] in initial_batch,
                        )
                    )
    for case in m["extra_cases"]:
        for functional in ("PBE", "PBE0"):
            m["tasks"].append(
                dict(case=case["name"], functional=functional, submit_initially=True)
            )
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    m["source_sha256"] = snapshot_package(snapshot)
    save_json(root / "manifest.json", m)
    locked_summary(root)
    return m


def get_case(m, name):
    cases = [c for s in m["series"] for c in s["cases"]] + m["extra_cases"]
    return next(c for c in cases if c["name"] == name)


def run(root, task_index, ranks, seconds, grace):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    _verify(root, m)
    task = m["tasks"][task_index]
    case = get_case(m, task["case"])
    with (root / f"task_{task_index}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            done = _launch(root, m, case, task["functional"], ranks, seconds, grace)
            result = dict(
                **task,
                complete=done,
                state="complete" if done else "checkpointed",
                job_id=os.environ.get("SLURM_JOB_ID"),
            )
        except Exception as exc:
            replace_json(
                root / f"task_{task_index}.json",
                dict(
                    **task,
                    complete=False,
                    state="failed",
                    job_id=os.environ.get("SLURM_JOB_ID"),
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        replace_json(root / f"task_{task_index}.json", result)
        locked_summary(root)
        print(json.dumps(result), flush=True)
        if not done:
            raise SystemExit(75)  # Saved checkpoint; never label it completed.


def validate_layout(root, ranks):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    _verify(root, m)
    c = m["layout_validation"]
    if not _launch(root, m, c, "PBE0", ranks, 3600, 1800):
        raise RuntimeError("Layout validation did not finish")
    old = json.loads(Path(m["layout_reference_report"]).read_text())
    new = json.loads((root / "cases" / c["name"] / "PBE0/report.json").read_text())
    delta = float(
        np.max(np.abs(np.array(old["eigenvalues_eV"]) - new["eigenvalues_eV"]))
    )
    result = dict(
        ranks=ranks,
        maximum_eigenvalue_difference_eV=delta,
        tolerance_eV=0.001,
        passed=delta < 0.001,
    )
    save_json(root / "layout_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Multi-node layout changes the spectrum by more than 1 meV")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--previous", required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--grace", type=int, default=3600)
    p = sub.add_parser("validate-layout")
    p.add_argument("--root", required=True)
    p.add_argument("--ranks", type=int, default=256)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.root, args.previous)
        print(
            json.dumps(
                {
                    "prepared_tasks": len(m["tasks"]),
                    "initial_tasks": sum(t["submit_initially"] for t in m["tasks"]),
                }
            )
        )
    elif args.command == "validate-layout":
        validate_layout(args.root, args.ranks)
    else:
        run(args.root, args.task, args.ranks, args.seconds, args.grace)


if __name__ == "__main__":
    main()
