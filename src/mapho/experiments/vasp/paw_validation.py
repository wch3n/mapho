"""Small native PAW replay acceptance study; all calculation files stay in scratch."""

import copy
import json
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
    read_pot,
    run,
    save_json,
    sha,
    write_incar,
)


def prepare(root, potential_root, installed_executable, bridge_executable):
    root, potential_root = Path(root).resolve(), Path(potential_root).resolve()
    manifest_path = root / "study.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    cases = []
    for name, atoms, datasets, nbands in (
        ("Si", bulk("Si", "diamond", a=5.43), {"Si": "Si"}, 12),
        ("LiF", bulk("LiF", "rocksalt", a=4.03), {"Li": "Li_sv", "F": "F"}, 16),
    ):
        reference = root / name / "reference"
        m = prepare_reference(
            reference,
            atoms,
            {s: potential_root / p / "POTCAR" for s, p in datasets.items()},
            encut=600.0,
            mesh=(2, 2, 2),
            nbands=nbands,
            algorithm="Exact",
        )
        control = root / name / "unpatched"
        control.mkdir()
        for f in ("POSCAR", "POTCAR", "INCAR", "KPOINTS"):
            shutil.copyfile(reference / f, control / f)
        cm = copy.deepcopy(m)
        cm["unpatched_control"] = True
        save_json(control / "manifest.json", cm)
        cases.append(name)
    save_json(
        manifest_path,
        dict(
            schema="mapho.vasp_paw_validation.v1",
            cases=cases,
            installed_executable=str(Path(installed_executable).resolve()),
            bridge_executable=str(Path(bridge_executable).resolve()),
            tolerance_eV=1e-4,
            shift_eV=0.25,
            purpose="operator interface validation, not accuracy/convergence benchmarks",
        ),
    )


def validate(root):
    root = Path(root).resolve()
    m = json.loads((root / "study.json").read_text())
    report = dict(schema=m["schema"], cases={}, complete=False, passed=False)

    # The installed NVHPC MPI wrapper cannot reliably start a singleton VASP
    # process. An explicit single-rank launcher also makes the layout auditable.
    def execute(folder, executable):
        status_path = folder / "run.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            inputs = json.loads((folder / "manifest.json").read_text())["inputs"]
            if status["status"] != "complete" or status["executable_sha256"] != sha(
                executable
            ):
                raise ValueError(
                    f"Cannot reuse incomplete or different executable run: {folder}"
                )
            if any(sha(folder / f) != digest for f, digest in inputs.items()):
                raise ValueError(f"Cannot reuse changed inputs: {folder}")
            read_bands(folder)
            return status
        return run(
            folder,
            executable,
            launcher=[
                "mpirun",
                "--mca",
                "pml",
                "ob1",
                "--mca",
                "btl",
                "self,vader",
                "-np",
                "1",
            ],
        )

    save_json(root / "validation.json", report)
    for name in m["cases"]:
        case = root / name
        ref, control = case / "reference", case / "unpatched"
        execute(control, m["installed_executable"])
        execute(ref, m["bridge_executable"])
        result = dict(
            unpatched_vs_bridge=compare(control, ref, tolerance_eV=m["tolerance_eV"])
        )
        report["cases"][name] = result
        save_json(root / "validation.json", report)
        operator = case / "operator.h5"
        export_operator(ref, operator)
        for tag, shift in (("replay", 0.0), ("shift", m["shift_eV"])):
            prepare_replay(operator, case / tag, shift_eV=shift)
            execute(case / tag, m["bridge_executable"])
            result[tag] = compare(ref, case / tag, shift, m["tolerance_eV"])
            save_json(root / "validation.json", report)
        # Negative control: moving V alone is not the PAW gauge shift c S.
        _, field, _, _ = read_pot(ref / "POT")
        prepare_replay(
            operator, case / "smooth_only_shift", potential=field + m["shift_eV"]
        )
        execute(case / "smooth_only_shift", m["bridge_executable"])
        result["smooth_only_shift"] = compare(
            ref, case / "smooth_only_shift", m["shift_eV"], m["tolerance_eV"]
        )
        # Independently recompute a frozen-charge PBE operator with unmodified
        # VASP at k points absent from the SCF mesh. This checks k independence
        # of exported ingredients and use of native generalized eigensolver.
        new_k = np.array([[0.13, 0.21, 0.07], [0.31, 0.11, 0.23], [0.43, 0.27, 0.17]])
        replay_k, charge_k = case / "new_k_replay", case / "new_k_native"
        prepare_replay(operator, replay_k, kpoints=new_k)
        charge_k.mkdir()
        for f in ("POSCAR", "POTCAR", "KPOINTS"):
            shutil.copyfile(replay_k / f, charge_k / f)
        shutil.copyfile(ref / "CHGCAR", charge_k / "CHGCAR")
        settings = json.loads((replay_k / "manifest.json").read_text())["settings"]
        settings.update(ICHARG=11)
        write_incar(charge_k / "INCAR", settings)
        save_json(
            charge_k / "manifest.json",
            dict(
                unpatched_control=True,
                settings=settings,
                inputs={
                    f: sha(charge_k / f)
                    for f in ("POSCAR", "POTCAR", "KPOINTS", "INCAR", "CHGCAR")
                },
            ),
        )
        execute(charge_k, m["installed_executable"])
        execute(replay_k, m["bridge_executable"])
        result["new_k_replay_vs_native_frozen_charge"] = compare(
            charge_k, replay_k, tolerance_eV=m["tolerance_eV"]
        )
        bands = read_bands(ref)["eigenvalues_eV"]
        nocc = int(json.loads((ref / "manifest.json").read_text())["nelect"] / 2)
        result["reference"] = dict(
            nkpoints=len(bands),
            nbands=bands.shape[1],
            vbm_eV=float(bands[:, nocc - 1].max()),
            cbm_eV=float(bands[:, nocc].min()),
        )
        result["passed"] = all(
            result[key]["passed"]
            for key in (
                "unpatched_vs_bridge",
                "replay",
                "shift",
                "new_k_replay_vs_native_frozen_charge",
            )
        )
        result["negative_control_detected"] = not result["smooth_only_shift"]["passed"]
        result["passed"] &= result["negative_control_detected"]
        report["cases"][name] = result
        save_json(root / "validation.json", report)
    report.update(
        complete=True, passed=all(c["passed"] for c in report["cases"].values())
    )
    save_json(root / "validation.json", report)
    if not report["passed"]:
        raise RuntimeError("PAW replay acceptance failed; see validation.json")
    return report
