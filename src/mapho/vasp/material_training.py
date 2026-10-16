"""Matched local/pointwise/3D-FNO and SiLU/GELU binary PAW experiments."""

from ..source_snapshot import snapshot_package, source_hashes, source_path
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from .paw import (
    prepare_reference,
    prepare_replay,
    write_incar,
    run,
    export_operator,
    read_bands,
    save_json,
    sha,
)
from .training_validation import LAUNCHER, parallel


def read_json(path):
    return json.loads(Path(path).read_text())


def native_checks(training):
    import torch
    from e3nn import o3
    from ase.io import read
    from .model import native_projector_transform

    torch.set_num_threads(1)
    root = Path(training)
    m = read_json(root / "manifest.json")
    source = Path(m["entries"]["train"][0]["baseline"]).parent
    dataset = read_json(Path(m["dataset"]) / "manifest.json")
    original = read_json(source / "manifest.json")
    atoms = read(source / "POSCAR")
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.47
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    with h5py.File(source / "operator.h5") as f:
        v = f["potential_eV"][:]
        d = f["paw/d"][:]
        q = f["paw/q"][:]
        channels = json.loads(f["paw"].attrs["channels_json"])
        types = f["paw/types"][:]
    potcars = {p["symbol"]: p["path"] for p in original["potentials"]}
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
            potcars,
            mesh=(1, 1, 1),
            encut=original["settings"]["ENCUT"],
            nbands=original["settings"]["NBANDS"],
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
            kz = 0.0 if dataset["material"] == "mos2" else 0.07
            (directory / "KPOINTS").write_text(
                f"atomic-density k independence\n2\nReciprocal\n0 0 0 1\n.13 .21 {kz} 1\n"
            )
        manifest["inputs"] = {
            f: sha(directory / f) for f in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
        }
        save_json(directory / "manifest.json", manifest)
        run(directory, dataset["executable"], launcher=LAUNCHER, timeout=1500)
        export_operator(directory, directory / "operator.h5")
        expected_d = d.copy()
        expected_q = q.copy()
        if name == "rotation":
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                for atom, kind in enumerate(types):
                    ls = channels[kind]
                    n = sum(2 * l + 1 for l in ls)
                    t = native_projector_transform(ls)
                    rep = o3.Irreps([(1, (l, (-1) ** l)) for l in ls]).D_from_matrix(
                        torch.tensor(rotation)
                    )
                    native = (t @ rep @ t.T).numpy()
                    expected_d[atom, :n, :n] = native @ d[atom, :n, :n] @ native.T
                    expected_q[atom, :n, :n] = native @ q[atom, :n, :n] @ native.T
            finally:
                torch.set_default_dtype(old)
        with h5py.File(directory / "operator.h5") as f:
            ve = float(np.abs(f["potential_eV"][:] - v).max())
            de = float(np.abs(f["paw/d"][:] - expected_d).max())
            qe = float(np.abs(f["paw/q"][:] - expected_q).max())
        checks[name] = dict(
            potential_max_eV=ve,
            onsite_max_eV=de,
            overlap_max=qe,
            passed=bool(ve < 1e-5 and de < 1e-5 and qe < 1e-10),
        )
    result = dict(checks=checks, passed=all(c["passed"] for c in checks.values()))
    save_json(root / "native_checks.json", result)
    if not result["passed"]:
        raise RuntimeError("Native projector convention check failed")


def bands_case(root, index):
    root = Path(root)
    m = read_json(root / "silu/manifest.json")
    entry = m["entries"]["test"][index]
    dataset = read_json(Path(m["dataset"]) / "manifest.json")
    executable = dataset["executable"]
    if sha(executable) != dataset["executable_sha256"]:
        raise ValueError("Executable changed")
    points = np.array([[0.0, 0.0, 0.0], [0.13, 0.21, 0.07], [0.31, 0.11, 0.23]])
    if dataset["material"] == "mos2":
        points[:, 2] = 0.0
    folder = root / "bands" / entry["name"]
    folder.mkdir(parents=True, exist_ok=False)
    target = Path(entry["target"])
    baseline = Path(entry["baseline"])
    spectra = {}
    variants = [
        "reference",
        "atomic",
        "silu_local",
        "silu_none",
        "silu_full",
        "gelu_none",
        "gelu_full",
    ]
    for variant in variants:
        kwargs = {}
        operator = target if variant == "reference" else baseline
        if variant not in ("reference", "atomic"):
            activation, mode = variant.split("_")
            with np.load(
                root / activation / mode / "predictions" / f"{entry['name']}.npz"
            ) as f:
                kwargs = dict(potential=f["potential"], d=f["d"])
        prepare_replay(operator, folder / variant, kpoints=points, **kwargs)
        run(folder / variant, executable, launcher=LAUNCHER, timeout=1500)
        spectra[variant] = read_bands(folder / variant)["eigenvalues_eV"]
    native = folder / "native"
    native.mkdir()
    for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS"):
        shutil.copyfile(folder / "reference" / name, native / name)
    shutil.copyfile(target.parent / "CHGCAR", native / "CHGCAR")
    settings = read_json(folder / "reference/manifest.json")["settings"]
    settings["ICHARG"] = 11
    write_incar(native / "INCAR", settings)
    save_json(
        native / "manifest.json",
        dict(
            unpatched_control=True,
            settings=settings,
            inputs={
                n: sha(native / n)
                for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
            },
        ),
    )
    run(native, executable, launcher=LAUNCHER, timeout=1500)
    spectra["native"] = read_bands(native)["eigenvalues_eV"]
    from .activation_ablation import band_metrics

    nocc = int(dataset["nelect"] / 2)
    metrics = {
        variant: band_metrics(e, spectra["native"], nocc, 4)
        for variant, e in spectra.items()
    }
    result = dict(
        name=entry["name"],
        kpoints=points.tolist(),
        nocc=nocc,
        frontier_half_window=4,
        metrics=metrics,
        reference_frontier_replay_passed=metrics["reference"]["frontier_max_eV"]
        < 0.001,
        eigenvalues_eV={k: v.tolist() for k, v in spectra.items()},
    )
    save_json(folder / "report.json", result)
    if not result["reference_frontier_replay_passed"]:
        raise RuntimeError("Reference replay exceeds 1 meV")


def execute(root):
    from .atomic_reference import prepare as prepare_atomic, execute as execute_atomic
    from .train import prepare as prepare_training

    root = Path(root).resolve()
    study = read_json(root / "study.json")
    dataset = Path(study["dataset"])
    if sha(dataset / "manifest.json") != study["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    if not read_json(dataset / "data_audit.json")["passed"]:
        raise ValueError("DFT audit failed")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for name, digest in study["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError("Training source changed")
    prepare_atomic(dataset, root / "atomic")
    execute_atomic(root / "atomic")
    shape = (32, 32, 64) if study["material"] == "mos2" else None
    m = prepare_training(dataset, root / "atomic", root / "silu", grid_shape=shape)
    gelu = root / "gelu"
    gelu.mkdir()
    (gelu / "logs").mkdir()
    (gelu / "cache").symlink_to(root / "silu/cache", target_is_directory=True)
    save_json(gelu / "manifest.json", dict(m, correction_activation="gelu"))
    subprocess_command = [
        sys.executable,
        "-u",
        "-m",
        "mapho.vasp.material_training",
        "native-checks",
        str(root / "silu"),
    ]
    parallel(root, [("native_checks", subprocess_command)], devices[:1])
    # The two activations must start with exactly the same parameters.
    import torch
    from .model import PAWModel
    from ..field_model import FieldModelConfig
    from .activation_ablation import parameters_sha

    initial = {}
    for mode in ("none", "full"):
        hashes = {}
        for activation in ("silu", "gelu"):
            torch.manual_seed(31)
            model = PAWModel(
                FieldModelConfig(**m["config"]),
                mode=mode,
                avg_neighbors=m["avg_neighbors"],
                correction_activation=activation,
                species_channels=m["species_channels"],
            )
            hashes[activation] = parameters_sha(model)
        if len(set(hashes.values())) != 1:
            raise ValueError("Activation initializations differ")
        initial[mode] = hashes
    save_json(
        root / "initialization_audit.json", dict(passed=True, parameter_sha256=initial)
    )
    tasks = [
        (act, mode)
        for act in ("silu", "gelu")
        for mode in ("local", "none", "full")
        if not (act == "gelu" and mode == "local")
    ]
    commands = [
        (
            act + "_" + mode,
            [
                sys.executable,
                "-u",
                "-m",
                "mapho.vasp.training_validation",
                "fit",
                str(root / act),
                mode,
                "--epochs",
                "300",
            ],
        )
        for act, mode in tasks
    ]
    parallel(root, commands, devices)
    parallel(
        root,
        [
            (
                f"bands_{i}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.vasp.material_training",
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
        read_json(root / "bands" / e["name"] / "report.json")
        for e in m["entries"]["test"]
    ]
    result = dict(
        complete=True,
        material=study["material"],
        job=os.environ.get("SLURM_JOB_ID"),
        models={
            act + "_" + mode: dict(
                training=read_json(root / act / mode / "training.json"),
                fields=read_json(root / act / mode / "test_fields.json"),
                frontier_mae_eV=float(
                    np.mean(
                        [
                            r["metrics"][act + "_" + mode]["frontier_mae_eV"]
                            for r in rows
                        ]
                    )
                ),
            )
            for act, mode in tasks
        },
        test_structures=len(rows),
        native_checks=read_json(root / "silu/native_checks.json"),
        scope="Same 300-epoch/seed31 budget and matched width/depth; 4 held-out structures; no test-based selection",
    )
    save_json(root / "completion.json", result)


def prepare(root, dataset):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    m = read_json(dataset / "manifest.json")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    study = dict(
        schema="mapho.vasp_material_training.v1",
        dataset=str(dataset),
        material=m["material"],
        dataset_manifest_sha256=sha(dataset / "manifest.json"),
        epochs=300,
        seed=31,
        modes=["local", "none", "full"],
        activations=["silu", "gelu"],
        source_sha256=source_hashes(snapshot),
        grid_shape=[32, 32, 64] if m["material"] == "mos2" else [32, 32, 32],
        slab_policy="3D periodic spectral operator; explicit height-mixing architecture reserved for separate comparison",
    )
    save_json(root / "study.json", study)
    return study


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("dataset")
    q = sub.add_parser("execute")
    q.add_argument("root")
    q = sub.add_parser("native-checks")
    q.add_argument("training")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("index", type=int)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.root, a.dataset)
    elif a.command == "execute":
        execute(a.root)
    elif a.command == "native-checks":
        native_checks(a.training)
    else:
        bands_case(a.root, a.index)
