"""Conditional Si PBE0 head learning with fixed reference exchange at each geometry.

The learned input remains geometry plus an atomic-density PBE baseline. The
complete eigenvalue comparison also consumes target-derived ACE exchange and
therefore does not establish structure-only hybrid inference.
"""

import json
from pathlib import Path
import shutil

from ase.build import bulk
import h5py
import numpy as np

from ...source_snapshot import snapshot_package
from ...vasp.ace import prepare_hybrid_replay
from ...vasp.paw import (
    export_operator,
    prepare_reference,
    run,
    save_json,
    sha,
    write_incar,
)
from ...vasp import train as training
from .hybrid_pilot import INPUTS, LAUNCHER, band_metrics
from .pbe0_convergence import (
    read_json,
    collect,
    verify_source,
    worker as reference_worker,
)


def structure(index):
    rng = np.random.default_rng(92000 + index)
    atoms = bulk("Si", "diamond", a=5.43)
    strain = rng.normal(0, 0.003, (3, 3))
    strain = 0.5 * (strain + strain.T)
    strain += np.eye(3) * rng.uniform(-0.01, 0.01)
    # Keep the principal strains inside the calibration interval.
    largest = np.max(np.abs(np.linalg.eigvalsh(strain)))
    strain *= min(1.0, 0.015 / max(largest, 1e-12))
    atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
    displacement = rng.uniform(-0.01, 0.01, 3)
    atoms.positions += np.array([displacement, -displacement])
    atoms.wrap()
    return atoms


def prepare_dataset(root, calibration, count=24):
    root, calibration = Path(root).resolve(), Path(calibration).resolve()
    gate = collect(calibration)
    if not gate["complete"] or not gate["passed"]:
        raise ValueError("PBE0 k-point convergence must pass before dataset generation")
    if count < 12 or int(count) != count:
        raise ValueError("Use at least twelve independent structures")
    cm = read_json(calibration / "manifest.json")
    if (
        sha(cm["executable"]) != cm["executable_sha256"]
        or sha(cm["potcar"]) != cm["potcar_sha256"]
    ):
        raise ValueError("Calibrated executable or PAW data changed")
    mesh = gate["selected_mesh"]
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    cases, operators = [], {}
    for index in range(count):
        name = f"sample_{index:03d}"
        case = root / name
        atoms = structure(index)
        for functional in ("PBE", "PBE0"):
            folder = case / functional.lower()
            m = prepare_reference(
                folder,
                atoms,
                {"Si": cm["potcar"]},
                mesh=(mesh,) * 3,
                encut=600.0,
                nbands=12,
                algorithm="Normal",
            )
            settings = m["settings"]
            settings.update(EDIFF=1e-10, NELM=240, NELMIN=12)
            if functional == "PBE0":
                settings.update(
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
                purpose="Conditional PBE0 local/PAW learning reference",
            )
            m["inputs"] = {key: sha(folder / key) for key in INPUTS}
            if functional == "PBE0":
                m["inputs"]["MAPHO_FOCK_EXPORT"] = sha(folder / "MAPHO_FOCK_EXPORT")
            save_json(folder / "manifest.json", m)
        shutil.copyfile(case / "pbe0/POSCAR", case / "POSCAR")
        cases.append(dict(directory=name, structure=name, mesh=mesh))
        operators[name + "/operator_components.h5"] = name + "/operator.h5"
    ids = np.random.default_rng(31).permutation(count).tolist()
    ntest = max(2, count // 6)
    nval = ntest
    partitions = dict(
        train=ids[nval + ntest :], validation=ids[:nval], test=ids[nval : nval + ntest]
    )
    m = dict(
        schema="mapho.si_pbe0_head_dataset.v1",
        cases=cases,
        count=count,
        natoms=2,
        executable=cm["executable"],
        executable_sha256=cm["executable_sha256"],
        potcar=cm["potcar"],
        potcar_sha256=cm["potcar_sha256"],
        calibration=str(calibration),
        calibration_manifest_sha256=sha(calibration / "manifest.json"),
        convergence_sha256=sha(calibration / "convergence.json"),
        mesh=mesh,
        source_sha256=snapshot_package(root / "source_snapshot/mapho"),
        replay_tolerance_eV=1e-4,
        hybrid_operators=operators,
        splits={
            split: [f"sample_{i:03d}/operator_components.h5" for i in indices]
            for split, indices in partitions.items()
        },
        correction_activation="gelu",
        modes=["local", "full"],
        inference_scope="Conditional PBE0 prediction with target-derived reference ACE exchange; same mesh only",
        strain_bound=0.015,
        displacement_component_bound_A=0.01,
        calibration_scope="Three representative geometries; common-k energies and local/PAW targets, not arbitrary-k convergence",
    )
    save_json(root / "manifest.json", m)
    atomic = root / "atomic"
    atomic.mkdir()
    save_json(
        atomic / "manifest.json",
        dict(dataset=str(root), dataset_manifest_sha256=sha(root / "manifest.json")),
    )
    return m


def worker(root, index):
    root = Path(root).resolve()
    m = verify_source(root)
    native = reference_worker(root, index)
    case = root / m["cases"][index]["directory"]
    name = case.name
    folder = root / "atomic" / name
    with h5py.File(case / "operator_components.h5") as f:
        shape = f["potential_eV"].shape
    baseline = prepare_reference(
        folder,
        structure(index),
        {"Si": m["potcar"]},
        mesh=(1, 1, 1),
        encut=600.0,
        nbands=12,
    )
    settings = baseline["settings"]
    settings.update(
        ICHARG=12,
        NELMDL=0,
        NELMIN=16,
        LWAVE=False,
        LCHARG=False,
        NGXF=shape[0],
        NGYF=shape[1],
        NGZF=shape[2],
    )
    write_incar(folder / "INCAR", settings)
    baseline["inputs"] = {key: sha(folder / key) for key in INPUTS}
    save_json(folder / "manifest.json", baseline)
    run(folder, m["executable"], launcher=LAUNCHER, timeout=1800)
    export_operator(folder, folder / "operator.h5")
    result = dict(
        complete=True,
        native_replay_passed=native["passed"],
        baseline_sha256=sha(folder / "operator.h5"),
        native_status_sha256=sha(case / "status.json"),
    )
    save_json(case / "data_complete.json", result)
    return result


def audit(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    rows = []
    for case in m["cases"]:
        folder = root / case["directory"]
        p = folder / "data_complete.json"
        if not p.exists():
            rows.append(dict(name=folder.name, passed=False, reason="pending"))
            continue
        data = read_json(p)
        if not data["complete"] or not data["native_replay_passed"]:
            raise ValueError("Native reference/replay failed")
        if data["baseline_sha256"] != sha(
            root / "atomic" / folder.name / "operator.h5"
        ) or data["native_status_sha256"] != sha(folder / "status.json"):
            raise ValueError("Dataset result changed")
        for name, digest in read_json(folder / "status.json")["output_sha256"].items():
            if sha(folder / name) != digest:
                raise ValueError("Native operator changed")
        rows.append(dict(name=folder.name, passed=True))
    result = dict(
        passed=all(row["passed"] for row in rows),
        rows=rows,
        dataset_manifest_sha256=sha(root / "manifest.json"),
    )
    save_json(root / "data_audit.json", result)
    if result["passed"]:
        save_json(
            root / "atomic/completion.json",
            dict(complete=True, dataset_manifest_sha256=sha(root / "manifest.json")),
        )
    return result


def prepare_training(root, dataset):
    dataset = Path(dataset).resolve()
    if not audit(dataset)["passed"]:
        raise ValueError(
            "Every reference and atomic baseline must pass before training"
        )
    return training.prepare(
        dataset,
        dataset / "atomic",
        root,
        target_kind="pbe0_components",
        correction_activation="gelu",
    )


def fit(root, mode, epochs=300):
    if mode not in ("local", "full"):
        raise ValueError("This pilot compares MAPHO-MACE and MAPHO-FNO")
    m = read_json(Path(root) / "manifest.json")
    if m["target_kind"] != "pbe0_components" or m["correction_activation"] != "gelu":
        raise ValueError("Expected the conditional GELU PBE0 training contract")
    training.train(root, mode, epochs=epochs)
    return training.evaluate(root, mode)


def evaluate_bands(root, mode):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    if m.get("target_kind") != "pbe0_components" or mode not in ("local", "full"):
        raise ValueError("Expected a conditional PBE0 model")
    dm = read_json(Path(m["dataset"]) / "manifest.json")
    if sha(dm["executable"]) != dm["executable_sha256"]:
        raise ValueError("Validated executable changed")
    model = training.load_model(root, mode, "cuda")
    rows = []
    for entry in m["entries"]["test"]:
        potential, d = training.predict(model, entry, "cuda")
        operator = entry["reference_exchange_operator"]
        if sha(operator) != entry["reference_exchange_operator_sha256"]:
            raise ValueError("Conditional reference exchange changed")
        with h5py.File(operator) as archive:
            reference = json.loads(archive.attrs["metadata_json"])["reference"]
        folder = root / mode / "bands" / entry["name"]
        prepare_hybrid_replay(operator, folder, potential=potential, d=d)
        run(folder, dm["executable"], launcher=LAUNCHER, timeout=162000)
        rows.append(dict(name=entry["name"], **band_metrics(reference, folder)))
    report = dict(
        mode=mode,
        conditional_reference_exchange=True,
        rows=rows,
        scope=m["prediction_scope"],
    )
    save_json(root / mode / "test_bands.json", report)
    return report
