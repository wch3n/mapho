"""Prepare reciprocal-spacing studies and run bounded native-QE SCF checks.

No torch/QEpy import in the worker. Pymatgen is needed only at preparation.
A mesh proposal is never a convergence certificate. The summary fails closed
when a mesh, functional, common-k comparison, or completed SCF is missing.
"""

from ..source_snapshot import snapshot_package, source_path
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from ase.io import read
from ase.io.espresso import read_fortran_namelist

from .supercell_reference import (
    HARTREE_EV,
    digest,
    read_spectrum,
    save_json,
    write_case,
)
from .files import save_directory


def replace_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_spectra(a, b, nocc, natoms, tolerance_eV=0.001):
    """Same geometry: compare frontier levels at identical k, with no shift fit.

    QE fixes the same periodic-potential convention at every mesh. Report raw
    level shifts as well as gap changes; never hide drift by fitting a shift.
    Sampling extrema on different meshes is a separate diagnostic/criterion.
    """
    ea, eb = np.asarray(a["eigenvalues_eV"]), np.asarray(b["eigenvalues_eV"])
    ka = np.asarray(a["kpoints_cartesian_2pi_over_alat"])
    kb = np.asarray(b["kpoints_cartesian_2pi_over_alat"])
    common = []
    for i, point in enumerate(ka):
        hit = np.flatnonzero(np.linalg.norm(kb - point, axis=1) < 1e-8)
        if len(hit) == 1:
            common.append((i, int(hit[0])))
    if not common:
        raise ValueError("No common k points: cannot certify band-edge convergence")
    if not any(np.linalg.norm(ka[i]) < 1e-8 for i, _ in common):
        raise ValueError("The Gamma anchor is missing")
    band_slice = slice(max(0, nocc - 4), min(ea.shape[1], eb.shape[1], nocc + 4))
    frontier = max(
        float(np.max(np.abs(ea[i, band_slice] - eb[j, band_slice]))) for i, j in common
    )
    gap = abs(a["sampled_gap_eV"] - b["sampled_gap_eV"])
    energy = (
        abs(a["total_energy_hartree"] - b["total_energy_hartree"]) * HARTREE_EV / natoms
    )
    checks = dict(
        common_k_frontier_max_eV=frontier,
        sampled_gap_change_eV=gap,
        total_energy_change_eV_per_atom=energy,
    )
    return dict(
        **checks,
        common_k_count=len(common),
        tolerance_eV=tolerance_eV,
        passed=bool(all(v <= tolerance_eV for v in checks.values())),
    )


def prepare(root, baseline, spacings=(0.8, 0.55, 0.4), tolerance_eV=0.001):
    from ..kpoints import mesh_from_spacing

    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    original = json.loads((baseline / "manifest.json").read_text())
    shutil.copytree(baseline / "pseudos", root / "pseudos")
    pseudos = original["pseudopotentials"]
    series = []
    for base in original["cases"]:
        if (
            base["info"]["pattern"] not in ("pristine", "longitudinal")
            or base["info"]["harmonic"] != 1
        ):
            continue
        atoms = read(
            baseline / "cases" / base["name"] / "PBE/qe.in", format="espresso-in"
        )
        proposals = [mesh_from_spacing(atoms, (spacings[0], s, s)) for s in spacings]
        if len(proposals) < 3 or len({tuple(p["mesh"]) for p in proposals}) != len(
            proposals
        ):
            raise ValueError(
                "Need at least three distinct progressively finer transverse meshes"
            )
        if any(
            np.any(np.array(b["mesh"]) < a["mesh"])
            for a, b in zip(proposals, proposals[1:])
        ):
            raise ValueError("Meshes must refine monotonically")
        finest = proposals[-1]
        # A separate longitudinal check, with the finest transverse spacing.
        long_spacing = finest["actual_spacing_inv_angstrom"][0] / 2 * (1 + 1e-12)
        proposals.append(
            mesh_from_spacing(atoms, (long_spacing, spacings[-1], spacings[-1]))
        )
        cases = []
        for i, proposal in enumerate(proposals):
            mesh = proposal["mesh"]
            name = base["name"] + "_k" + "x".join(map(str, mesh))
            case = write_case(
                root, name, atoms, base["info"], pseudos, mesh=tuple(mesh)
            )
            case["mesh_proposal"] = proposal
            case["role"] = (
                "longitudinal_check"
                if i == len(proposals) - 1
                else "transverse_refinement"
            )
            if mesh == base["kmesh"]:
                case["reuse_candidate"] = str(baseline / "cases" / base["name"])
                case["reuse_input_hashes"] = {
                    f: base["inputs"][f]["sha256"] for f in ("PBE", "PBE0")
                }
            cases.append(case)
        series.append(dict(name=base["name"], cases=cases))
    pilot_base = original["pilot"]
    pilot_atoms = read(
        baseline / "cases" / pilot_base["name"] / "PBE/qe.in", format="espresso-in"
    )
    pilot = write_case(
        root,
        "restart_validation",
        pilot_atoms,
        {"role": "clean stop/restart validation"},
        pseudos,
        mesh=(2, 2, 2),
        nbnd=8,
        conv_thr=1e-10,
    )
    pilot["reference_directory"] = str(baseline / "cases" / pilot_base["name"])
    snapshot = root / "source_snapshot" / "mapho"
    snapshot.mkdir(parents=True)
    hashes = snapshot_package(snapshot)
    manifest = dict(
        schema="mapho.kpoint_convergence.v1",
        series=series,
        pilot=pilot,
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "manifest.json"),
        pseudopotentials=pseudos,
        executable=str(baseline / "qe_source/bin/pw.x"),
        executable_sha256=(baseline / "pw.sha256").read_text().split()[0],
        source_sha256=hashes,
        tolerance_eV=tolerance_eV,
        selected_mesh=None,
        training_authorized=False,
        max_allocations_per_series=4,
        max_seconds_per_launch=5400,
        scope="k/q convergence at fixed pseudopotentials and 100/400 Ry cutoffs",
        criteria=[
            "two successive transverse comparisons plus longitudinal doubling",
            "PBE and PBE0: 1 meV common-k frontier levels and sampled gap",
            "PBE and PBE0: 1 meV/atom total energy",
        ],
        limitations=[
            "This study does not establish plane-wave cutoff convergence.",
            "Native field convergence must be checked before field training.",
            "Representative pristine and longitudinal geometries; additional configurations require spot checks.",
        ],
    )
    save_json(root / "manifest.json", manifest)
    summarize(root)
    return manifest


def summarize(root):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    result = dict(
        schema="mapho.kpoint_convergence_summary.v1",
        updated_utc=datetime.now(timezone.utc).isoformat(),
        series=[],
        all_spectral_checks_passed=False,
        training_ready=False,
        training_hold_reason="Reference sampling and field checks must pass before training.",
    )
    for series in m["series"]:
        row = dict(name=series["name"], functionals={}, selected_mesh=None)
        for functional in ("PBE", "PBE0"):
            cases = series["cases"]
            pairs = [(i, i + 1) for i in range(len(cases) - 2)] + [
                (len(cases) - 2, len(cases) - 1)
            ]
            checks = []
            for a, b in pairs:
                pa = root / "cases" / cases[a]["name"] / functional / "report.json"
                pb = root / "cases" / cases[b]["name"] / functional / "report.json"
                entry = dict(coarse=cases[a]["kmesh"], fine=cases[b]["kmesh"])
                if pa.exists() and pb.exists():
                    try:
                        entry.update(
                            compare_spectra(
                                json.loads(pa.read_text()),
                                json.loads(pb.read_text()),
                                cases[a]["nocc"],
                                cases[a]["natoms"],
                                m["tolerance_eV"],
                            )
                        )
                        entry["state"] = (
                            "passed" if entry["passed"] else "not_converged"
                        )
                    except ValueError as exc:
                        entry.update(state="invalid", passed=False, error=str(exc))
                else:
                    entry.update(state="pending", passed=False)
                checks.append(entry)
            row["functionals"][functional] = dict(
                checks=checks, passed=all(c["passed"] for c in checks)
            )
        row["spectral_checks_passed"] = all(
            v["passed"] for v in row["functionals"].values()
        )
        if row["spectral_checks_passed"]:
            row["selected_mesh"] = series["cases"][-1]["kmesh"]
        result["series"].append(row)
    result["all_spectral_checks_passed"] = all(
        s["spectral_checks_passed"] for s in result["series"]
    )
    # Serialize concurrent writers, and compute the summary within the lock
    # in the public command / worker caller, to avoid stale overwrites.
    replace_json(root / "summary.json", result)
    return result


def locked_summary(root):
    with (Path(root) / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return summarize(root)


def _verify(root, m):
    if digest(Path(m["baseline"]) / "manifest.json") != m["baseline_manifest_sha256"]:
        raise RuntimeError("Baseline manifest changed")
    for name, expected in m["source_sha256"].items():
        if digest(source_path(name)) != expected:
            raise RuntimeError(f"Snapshot changed: {name}")
    if digest(m["executable"]) != m["executable_sha256"]:
        raise RuntimeError("QE executable changed")
    for symbol, meta in m["pseudopotentials"].items():
        if digest(root / "pseudos" / f"{symbol}.upf") != meta["sha256"]:
            raise RuntimeError("Pseudopotential changed")


def equivalent_inputs(old_path, new_path, coordinate_atol=1e-8):
    """Require the same physics; tolerate only coordinate serialization roundoff.

    Namelist values and every card other than Cartesian atomic coordinates
    remain exact. This is not a geometry-equivalence or symmetry search.
    """
    with Path(old_path).open() as f:
        old, old_cards = read_fortran_namelist(f)
    with Path(new_path).open() as f:
        new, new_cards = read_fortran_namelist(f)
    for data in (old, new):
        for key in ("outdir", "pseudo_dir", "max_seconds", "restart_mode", "prefix"):
            data["control"].pop(key, None)
    if old != new or len(old_cards) != len(new_cards):
        raise RuntimeError("Reuse candidate differs in physical settings/cards")
    remaining = 0
    max_roundoff = 0.0
    for a, b in zip(old_cards, new_cards):
        if remaining:
            aa, bb = a.split(), b.split()
            if len(aa) != len(bb) or aa[0] != bb[0] or aa[4:] != bb[4:]:
                raise RuntimeError("Reuse candidate differs in species/constraints")
            error = float(
                np.max(np.abs(np.asarray(aa[1:4], float) - np.asarray(bb[1:4], float)))
            )
            if not np.isfinite(error) or error > coordinate_atol:
                raise RuntimeError("Reuse candidate differs in atomic coordinates")
            max_roundoff = max(max_roundoff, error)
            remaining -= 1
        else:
            if a != b:
                raise RuntimeError("Reuse candidate differs in physical cards")
            if a.lower() == "atomic_positions angstrom":
                remaining = int(old["system"]["nat"])
    if remaining:
        raise RuntimeError("Truncated atomic coordinates")
    return max_roundoff


def _reuse(root, case, functional):
    """Use only completed, hashed native references for identical settings."""
    if "reuse_candidate" not in case:
        return False
    source = Path(case["reuse_candidate"]) / functional
    rp = source / "report.json"
    if not rp.exists():
        return False
    if digest(source / "qe.in") != case["reuse_input_hashes"][functional]:
        raise RuntimeError("Baseline input changed")
    geometry_roundoff = equivalent_inputs(
        source / "qe.in", root / "cases" / case["name"] / functional / "qe.in"
    )
    old = json.loads(rp.read_text())
    xml = save_directory(source) / "data-file-schema.xml"
    if digest(xml) != old["xml_sha256"]:
        raise RuntimeError("Baseline XML changed")
    report = read_spectrum(
        xml, source / "qe.log", hybrid=functional == "PBE0", nocc=case["nocc"]
    )
    if len(report["weights"]) != math.prod(case["kmesh"]):
        raise RuntimeError("Baseline k-point count mismatch")
    destination = root / "cases" / case["name"] / functional
    report.update(
        reused_from=str(source),
        xml_sha256=digest(xml),
        source_report_sha256=digest(rp),
        geometry_roundoff_angstrom=geometry_roundoff,
    )
    save_json(destination / "report.json", report)
    # An immutable native save can seed the paired functional without copying
    # all native wavefunctions until actually needed.
    return True


def _launch(root, m, case, functional, ranks, seconds, grace_seconds=900):
    folder = root / "cases" / case["name"] / functional
    original = folder / "qe.in"
    if digest(original) != case["inputs"][functional]["sha256"]:
        raise RuntimeError("Prepared input changed")
    if (folder / "report.json").exists() or _reuse(root, case, functional):
        return True
    state_path = folder / "state.json"
    state = (
        json.loads(state_path.read_text()) if state_path.exists() else {"attempts": []}
    )
    if state.get("state") in ("failed", "running"):
        raise RuntimeError(
            "Prior failed attempt requires inspection; refusing silent retry"
        )
    restart = state.get("state") == "checkpointed"
    scratch = folder / "scratch"
    if not restart and not state["attempts"] and functional == "PBE0":
        pbe_folder = folder.parent / "PBE"
        pbe_report = json.loads((pbe_folder / "report.json").read_text())
        pbe_save = save_directory(Path(pbe_report.get("reused_from", str(pbe_folder))))
        scratch.mkdir(exist_ok=True)
        shutil.copytree(pbe_save, save_directory(folder))
    sequence = len(state["attempts"]) + 1
    input_path = folder / f"attempt_{sequence:02d}.in"
    log_path = folder / f"attempt_{sequence:02d}.log"
    text = original.read_text()
    text = re.sub(
        r"(max_seconds\s*=\s*)[0-9.]+", lambda match: match[1] + str(seconds), text
    )
    text = text.replace(
        "&CONTROL",
        "&CONTROL\n   restart_mode = 'restart',"
        if restart
        else "&CONTROL\n   restart_mode = 'from_scratch',",
        1,
    )
    input_path.write_text(text)
    # Four pools are legal for all planned meshes (4, 9, 16, 32 points).
    bands = 4 if functional == "PBE0" and ranks >= 32 else 1
    command = [
        "mpirun",
        "-np",
        str(ranks),
        "--bind-to",
        "core",
        m["executable"],
        "-nk",
        "4",
        "-nb",
        str(bands),
        "-ndiag",
        "1",
        "-in",
        str(input_path),
    ]
    attempt = dict(
        index=sequence,
        job_id=os.environ.get("SLURM_JOB_ID"),
        restart=restart,
        command=command,
        input_sha256=digest(input_path),
        requested_seconds=seconds,
        state="running",
    )
    state["attempts"].append(attempt)
    state["state"] = "running"
    replace_json(state_path, state)
    start = time.monotonic()
    try:
        with log_path.open("x") as log:
            proc = subprocess.run(
                command,
                cwd=folder,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds + grace_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        attempt.update(
            state="failed",
            detail="Launcher timeout; no clean checkpoint certified",
            wall_seconds=time.monotonic() - start,
        )
        state["state"] = "failed"
        replace_json(state_path, state)
        raise RuntimeError(attempt["detail"]) from exc
    attempt.update(returncode=proc.returncode, wall_seconds=time.monotonic() - start)
    xml = save_directory(folder) / "data-file-schema.xml"
    try:
        report = read_spectrum(
            xml, log_path, hybrid=functional == "PBE0", nocc=case["nocc"]
        )
        if proc.returncode != 0:
            raise RuntimeError("QE returned a nonzero status despite convergence text")
        if len(report["weights"]) != math.prod(case["kmesh"]):
            raise RuntimeError("Wrong k-point count")
        report.update(xml_sha256=digest(xml), mpi_command=command, attempts=sequence)
        save_json(folder / "report.json", report)
        attempt["state"] = state["state"] = "complete"
        replace_json(state_path, state)
        return True
    except (RuntimeError, OSError, ValueError) as exc:
        log = log_path.read_text(errors="replace")
        # Native clean max_seconds stops have a restart marker and native save.
        # A hard Slurm timeout, numerical error, or exhausted SCF is never a checkpoint.
        clean_stop = (
            "Maximum CPU time exceeded" in log
            or "maximum CPU time exceeded" in log
            or "Maximum wall time exceeded" in log
        ) and "JOB DONE." in log
        restart_files = list(scratch.glob("*restart*"))
        valid = (
            proc.returncode == 0 and clean_stop and xml.exists() and bool(restart_files)
        )
        attempt["state"] = state["state"] = "checkpointed" if valid else "failed"
        attempt["detail"] = str(exc)
        attempt["restart_files"] = [str(p) for p in restart_files]
        replace_json(state_path, state)
        if not valid:
            raise
        return False


def worker(root, index, ranks, allocation=1):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    _verify(root, m)
    series = m["series"][index]
    with (root / f"worker_{index}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        complete = True
        # Finish the inexpensive PBE ladder before starting hybrid refinement.
        for functional in ("PBE", "PBE0"):
            for case in series["cases"]:
                remaining = 6300 - int(time.monotonic() - start)
                if remaining < 600:
                    complete = False
                    break
                done = _launch(
                    root,
                    m,
                    case,
                    functional,
                    ranks,
                    min(m["max_seconds_per_launch"], remaining),
                )
                locked_summary(root)
                if not done:
                    complete = False
                    break
            if not complete:
                break
        status = dict(
            series=series["name"],
            allocation=allocation,
            complete=complete,
            job_id=os.environ.get("SLURM_JOB_ID"),
        )
        if not complete and allocation < m["max_allocations_per_series"]:
            current = os.environ.get("SLURM_JOB_ID")
            if not current:
                raise RuntimeError("Automatic continuation requires Slurm")
            cmd = [
                "sbatch",
                "--parsable",
                f"--dependency=afterany:{current}",
                str(root / "worker.sbatch"),
                str(root),
                str(index),
                str(allocation + 1),
            ]
            status["continuation_job"] = subprocess.check_output(cmd, text=True).strip()
        elif not complete:
            status["state"] = "allocation_budget_exhausted_unverified"
        else:
            status["state"] = "calculations_complete_see_summary_for_convergence"
        replace_json(root / f"worker_{index}.json", status)
        print(json.dumps(status), flush=True)


def validate_restart(root, ranks=8):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    _verify(root, m)
    case = m["pilot"]
    if not _launch(root, m, case, "PBE", ranks, 600):
        raise RuntimeError("Primitive PBE validation timed out")
    if _launch(root, m, case, "PBE0", ranks, 1):
        raise RuntimeError("Validation did not exercise a checkpoint")
    if not _launch(root, m, case, "PBE0", ranks, 1200):
        raise RuntimeError("Primitive PBE0 restart failed to finish")
    result = {"passed": True, "maximum_difference_eV": {}}
    for functional in ("PBE", "PBE0"):
        old = json.loads(
            (Path(case["reference_directory"]) / functional / "report.json").read_text()
        )
        new = json.loads(
            (root / "cases" / case["name"] / functional / "report.json").read_text()
        )
        error = float(
            np.max(np.abs(np.array(old["eigenvalues_eV"]) - new["eigenvalues_eV"]))
        )
        result["maximum_difference_eV"][functional] = error
        result["passed"] &= error < 0.001
    save_json(root / "restart_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Restart changes the primitive spectrum by more than 1 meV")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--spacings", type=float, nargs="+", default=[0.8, 0.55, 0.4])
    p.add_argument("--tolerance-ev", type=float, default=0.001)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--ranks", type=int, default=128)
    p.add_argument("--allocation", type=int, default=1)
    p = sub.add_parser("validate-restart")
    p.add_argument("--root", required=True)
    p.add_argument("--ranks", type=int, default=8)
    p = sub.add_parser("summarize")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.root, args.baseline, args.spacings, args.tolerance_ev)
        print(f"Prepared {len(m['series'])} geometry series")
    elif args.command == "validate-restart":
        validate_restart(args.root, args.ranks)
    elif args.command == "run":
        worker(args.root, args.index, args.ranks, args.allocation)
    else:
        print(json.dumps(locked_summary(args.root), indent=2))


if __name__ == "__main__":
    main()
