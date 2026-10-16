"""Controlled GELU versus SiLU test using existing Si16 and primitive references."""

from ..source_snapshot import snapshot_package, source_hashes
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from .paw import prepare_replay, read_bands, run, sha, save_json
from .training_validation import LAUNCHER, KS, parallel


def read_json(path):
    return json.loads(Path(path).read_text())


def parameters_sha(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare(root, baseline, primitive):
    import torch
    from .model import PAWModel
    from ..field_model import FieldModelConfig

    root, baseline, primitive = map(
        lambda p: Path(p).resolve(), (root, baseline, primitive)
    )
    if (
        not read_json(baseline / "training_audit.json")["passed"]
        or not read_json(primitive / "validation_audit.json")["passed"]
    ):
        raise ValueError("Validated SiLU training and primitive references required")
    old = read_json(baseline / "manifest.json")
    if old.get("correction_activation", "silu") != "silu":
        raise ValueError("Expected SiLU baseline")
    for mode in ("none", "full"):
        t = read_json(baseline / mode / "training.json")
        if t["epochs"] != 300 or t["seed"] != 31:
            raise ValueError("Expected matched seed31/300-epoch controls")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "cache").symlink_to(baseline / "cache", target_is_directory=True)
    manifest = dict(old, correction_activation="gelu")
    save_json(root / "manifest.json", manifest)
    initial = {}
    for mode in ("none", "full"):
        states = {}
        for activation in ("silu", "gelu"):
            torch.manual_seed(31)
            model = PAWModel(
                FieldModelConfig(**old["config"]),
                mode=mode,
                avg_neighbors=old["avg_neighbors"],
                correction_activation=activation,
            )
            states[activation] = parameters_sha(model)
        if states["silu"] != states["gelu"]:
            raise ValueError("Initial weights differ")
        initial[mode] = states
    reference_files = {}
    for e in old["entries"]["test"]:
        for path in (baseline / "bands" / e["name"] / "native").glob("*"):
            if path.name in ("EIGENVAL", "vasprun.xml", "manifest.json", "run.json"):
                reference_files[str(path)] = sha(path)
    for path in (primitive / "bands/native_16").glob("*"):
        if path.name in ("EIGENVAL", "vasprun.xml", "manifest.json", "run.json"):
            reference_files[str(path)] = sha(path)
    for path in (
        primitive / "references/atomic/operator.h5",
        primitive / "references/scf_16/operator.h5",
        primitive / "manifest.json",
        primitive / "results.json",
        baseline / "completion.json",
    ):
        reference_files[str(path)] = sha(path)
    cache_hashes_match = all(
        sha(root / "cache" / f"{e['name']}.npz") == e["cache_sha256"]
        for entries in old["entries"].values()
        for e in entries
    )
    if not cache_hashes_match:
        raise ValueError("Source training cache changed")
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    study = dict(
        schema="mapho.vasp_activation_ablation.v1",
        baseline=str(baseline),
        primitive=str(primitive),
        baseline_manifest_sha256=sha(baseline / "manifest.json"),
        baseline_checkpoints={
            mode: sha(baseline / mode / "model.pt") for mode in ("none", "full")
        },
        modes=["none", "full"],
        activation="gelu",
        gelu_approximation="none",
        epochs=300,
        seed=31,
        only_change="SiLU to exact GELU after each of the two correction-layer sums; MACE and radial-filter SiLU unchanged",
        unchanged="identical split, cached native points, input scaling, width/depth, initialization, optimizer, learning-rate scheduler, data order, loss and checkpoint selection",
        initial_parameter_sha256=initial,
        reference_sha256=reference_files,
        source_sha256=source_hashes(snapshot),
        validation="Four held-out Si16 structures plus the existing primitive X-Gamma-L path; no test-based selection",
        new_dft_scf_calculations=0,
    )
    save_json(root / "study.json", study)
    return study


def band_metrics(e, reference, nocc, half_window):
    delta = e - reference
    window = slice(nocc - half_window, nocc + half_window)
    gap = float(e[:, nocc].min() - e[:, nocc - 1].max())
    refgap = float(reference[:, nocc].min() - reference[:, nocc - 1].max())
    return dict(
        frontier_mae_eV=float(abs(delta[:, window]).mean()),
        frontier_max_eV=float(abs(delta[:, window]).max()),
        all_band_mae_eV=float(abs(delta).mean()),
        sampled_gap_eV=gap,
        sampled_gap_error_eV=gap - refgap,
    )


def band_case(root, mode, index):
    root = Path(root)
    s = read_json(root / "study.json")
    m = read_json(root / "manifest.json")
    dataset = read_json(Path(m["dataset"]) / "manifest.json")
    if sha(dataset["executable"]) != dataset["executable_sha256"]:
        raise ValueError("Executable changed")
    if index == "primitive":
        import torch
        from .train import load_model, predict

        torch.set_num_threads(1)
        original = Path(s["primitive"])
        baseline = original / "references/atomic/operator.h5"
        entry = dict(
            baseline=str(baseline), baseline_sha256=s["reference_sha256"][str(baseline)]
        )
        v, d = predict(load_model(root, mode, "cuda"), entry, "cuda")
        folder = root / "primitive" / mode
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / "prediction.npz", potential=v, d=d)
        pm = read_json(original / "manifest.json")
        points = np.array(pm["path"]["kpoints"])
        refdir = original / "bands/native_16"
        nocc, half_window = 4, 2
        with h5py.File(original / "references/scf_16/operator.h5") as f:
            fields = dict(
                field_rmse_eV=float(np.mean((v - f["potential_eV"][:]) ** 2) ** 0.5),
                onsite_rmse_eV=float(np.mean((d - f["paw/d"][:]) ** 2) ** 0.5),
            )
    else:
        entry = m["entries"]["test"][int(index)]
        baseline = Path(entry["baseline"])
        if sha(baseline) != entry["baseline_sha256"]:
            raise ValueError("Atomic input changed")
        with np.load(root / mode / "predictions" / f"{entry['name']}.npz") as f:
            v, d = f["potential"], f["d"]
        folder = root / "bands" / entry["name"] / mode
        folder.mkdir(parents=True, exist_ok=False)
        points = KS
        refdir = Path(s["baseline"]) / "bands" / entry["name"] / "native"
        nocc, half_window = 32, 4
        fields = None
    for filename in ("EIGENVAL", "vasprun.xml"):
        path = refdir / filename
        if sha(path) != s["reference_sha256"][str(path)]:
            raise ValueError("Native reference changed")
    native = read_bands(refdir)
    if not np.allclose(native["kpoints"], points, atol=1e-7, rtol=0):
        raise ValueError("Native path differs")
    prepare_replay(baseline, folder / "vasp", potential=v, d=d, kpoints=points)
    run(folder / "vasp", dataset["executable"], launcher=LAUNCHER, timeout=1200)
    prediction = read_bands(folder / "vasp")
    if not np.allclose(prediction["kpoints"], points, atol=1e-7, rtol=0):
        raise ValueError("Prediction path differs")
    e = prediction["eigenvalues_eV"]
    ref = native["eigenvalues_eV"]
    if e.shape != ref.shape:
        raise ValueError("Band dimensions differ")
    result = dict(
        mode=mode,
        activation="gelu",
        case=index,
        fields=fields,
        metrics=band_metrics(e, ref, nocc, half_window),
        native_reference=str(refdir),
        kpoints=points.tolist(),
        eigenvalues_eV=e.tolist(),
    )
    save_json(folder / "report.json", result)


def summarize(root):
    root = Path(root)
    s = read_json(root / "study.json")
    m = read_json(root / "manifest.json")
    old = read_json(Path(s["baseline"]) / "completion.json")
    primitive = read_json(Path(s["primitive"]) / "results.json")
    comparison = {}
    for mode in s["modes"]:
        reports = [
            read_json(root / "bands" / e["name"] / mode / "report.json")
            for e in m["entries"]["test"]
        ]
        fields = read_json(root / mode / "test_fields.json")
        p = read_json(root / "primitive" / mode / "report.json")
        t = read_json(root / mode / "training.json")
        comparison[mode] = dict(
            silu=dict(
                si16_fields=old["fields"][mode],
                si16_bands=old["bands"][mode],
                primitive_bands=primitive["metrics"][mode],
                primitive_fields=primitive["fields"][mode],
            ),
            gelu=dict(
                si16_fields=fields,
                si16_bands={
                    k: float(np.mean([x["metrics"][k] for x in reports]))
                    for k in ("frontier_mae_eV", "all_band_mae_eV")
                },
                primitive_bands=p["metrics"],
                primitive_fields=p["fields"],
                training=t,
            ),
        )
    result = dict(
        complete=True,
        job=os.environ.get("SLURM_JOB_ID"),
        comparison=comparison,
        baseline=str(Path(s["baseline"])),
        primitive_reference=str(Path(s["primitive"])),
        scope="One paired seed, fixed 300-epoch budget, only correction-layer activations changed",
        energy_alignment="none; every band error uses the same native VASP energy gauge",
    )
    save_json(root / "completion.json", result)
    return result


def execute(root):
    root = Path(root).resolve()
    s = read_json(root / "study.json")
    m = read_json(root / "manifest.json")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for path, expected in s["reference_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Frozen reference changed")
    commands = [
        (
            "fit_" + mode,
            [
                sys.executable,
                "-u",
                "-m",
                "mapho.vasp.training_validation",
                "fit",
                str(root),
                mode,
                "--epochs",
                str(s["epochs"]),
            ],
        )
        for mode in s["modes"]
    ]
    parallel(root, commands, devices)
    commands = []
    for index in [str(i) for i in range(len(m["entries"]["test"]))] + ["primitive"]:
        for mode in s["modes"]:
            commands.append(
                (
                    index + "_" + mode,
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "mapho.vasp.activation_ablation",
                        "bands",
                        str(root),
                        mode,
                        index,
                    ],
                )
            )
    parallel(root, commands, devices)
    summarize(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("baseline")
    q.add_argument("primitive")
    for name in ("execute", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("root")
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("mode", choices=["none", "full"])
    q.add_argument("index")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.root, a.baseline, a.primitive)
    elif a.command == "bands":
        band_case(a.root, a.mode, a.index)
    elif a.command == "summarize":
        summarize(a.root)
    else:
        execute(a.root)
