"""Fit overlap-scaled equivariant PAW heads with frozen field models.

Only training atoms determine scaling/centering; validation selects checkpoints.
Native validation precedes any new test prediction. Original runs are immutable.
"""

from ...vasp.checkpoints import (
    read_json,
    check_hash,
    load_base_for_repair,
    load_repaired_model,
)

from ...source_snapshot import snapshot_package, source_hashes, source_path
import json
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import torch

from ...field_model import mace_graph
from ...vasp.model import atomic_inputs
from ...vasp.paw import (
    sha,
    save_json,
    prepare_replay,
    run,
    read_bands,
    write_incar,
    export_operator,
)
from ...vasp.train import paw_contract, predict
from ...vasp.paw_conditioning import (
    ConditionedPAWModel,
    overlap_coordinates,
    scalar_projection,
)
from .training_validation import parallel, LAUNCHER, KS

LABELS = ("silu_local", "silu_none", "silu_full", "gelu_none", "gelu_full")


def prepare(root, training, refine_from=None):
    root, training = Path(root).resolve(), Path(training).resolve()
    m = read_json(training / "silu/manifest.json")
    dataset = read_json(Path(m["dataset"]) / "manifest.json")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    models = {}
    for label in LABELS:
        activation, mode = label.split("_")
        base = training / activation
        models[label] = dict(
            root=str(base),
            mode=mode,
            manifest_sha256=sha(base / "manifest.json"),
            checkpoint_sha256=sha(base / mode / "model.pt"),
        )
    extra = {}
    if refine_from is not None:
        previous = Path(refine_from).resolve()
        previous_manifest = read_json(previous / "manifest.json")
        if previous_manifest["models"] != models:
            raise ValueError("Refinement base models differ")
        extra = dict(
            optimizer="lbfgs",
            refine_from=str(previous),
            parent_manifest_sha256=sha(previous / "manifest.json"),
            parent_checkpoints={
                label: sha(previous / label / "model.pt") for label in LABELS
            },
        )
    save_json(
        root / "manifest.json",
        dict(
            schema="mapho.vasp_paw_repair.v1",
            training=str(training),
            models=models,
            **extra,
            entries=m["entries"],
            executable=dataset["executable"],
            executable_sha256=dataset["executable_sha256"],
            nocc=int(dataset["nelect"] / 2),
            seed=31,
            max_steps=100 if refine_from else 4000,
            validation_interval=1 if refine_from else 25,
            learning_rate=0.001,
            early_stop_patience=20 if refine_from else 750,
            minimum_steps=40 if refine_from else 1500,
            overlap_relative_floor=1e-8,
            normalization="training-only invariant mean and species RMS in |Q|^-1/2 coordinates",
            objective="equal-species normalized encoded matrix MSE",
            scope="Frozen original MACE and smooth potential, new equivariant PAW heads only",
            split_policy="Original train/validation/test split; known test failures motivated repair; no test fitting or selection",
            thresholds=dict(
                same_operator_replay_max_eV=0.001,
                spectral_all_band_max_eV=0.25,
                minimum_sampled_gap_eV=0.0,
            ),
            source_sha256=source_hashes(snapshot),
        ),
    )


def prepare_evaluation(root, trained):
    """New evaluation snapshot; fitted artifacts and their contracts stay frozen."""
    root, trained = Path(root).resolve(), Path(trained).resolve()
    m = read_json(trained / "manifest.json")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    for label in LABELS:
        (root / label).mkdir()
        for name in ("model.pt", "training.json"):
            (root / label / name).symlink_to(trained / label / name)
    (root / "validation_summary.json").symlink_to(trained / "validation_summary.json")
    m.update(
        trained_root=str(trained),
        trained_manifest_sha256=sha(trained / "manifest.json"),
        trained_checkpoints={
            label: sha(trained / label / "model.pt") for label in LABELS
        },
        field_reproducibility_tolerance_eV=1e-5,
        evaluation_note="Same frozen checkpoints; float32 GPU field replay compared within 1e-5 eV roundoff, not bitwise equality",
        source_sha256=source_hashes(snapshot),
    )
    save_json(root / "manifest.json", m)


def gather(base, entries, device):
    """No dense field decoder; raw PAW labels stay float64 until transformed."""
    rows = []
    reference_q = {}
    channels = {}
    for e in entries:
        check_hash(e["baseline"], e["baseline_sha256"])
        check_hash(e["target"], e["target_sha256"])
        with h5py.File(e["baseline"]) as b, h5py.File(e["target"]) as t:
            contract, _ = paw_contract(b)
            numbers = b["atomic_numbers"][:]
            for key in (
                "cell_A",
                "positions_A",
                "atomic_numbers",
                "paw/q",
                "paw/types",
            ):
                if not np.allclose(b[key][:], t[key][:], atol=1e-10, rtol=0):
                    raise ValueError("Target/baseline mismatch")
            if paw_contract(t)[0] != contract:
                raise ValueError("Target PAW channels differ")
            inputs = atomic_inputs(
                numbers, b["positions_A"][:], b["cell_A"][:], np.zeros((2, 2, 2))
            )
            graph = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in mace_graph(inputs, base.config).items()
            }
            with torch.no_grad():
                features = base.local.backbone(
                    graph, training=False, compute_force=False
                )["node_feats"].detach()
            dd = t["paw/d"][:] - b["paw/d"][:]
            q = b["paw/q"][:]
            row = dict(name=e["name"], parts={})
            for z, ls in contract.items():
                n = sum(2 * l + 1 for l in ls)
                ids = np.where(numbers == z)[0]
                key = str(z)
                if key in channels and channels[key] != ls:
                    raise ValueError("Species channels changed")
                channels[key] = ls
                if key not in reference_q:
                    reference_q[key] = q[ids[0], :n, :n]
                if not np.allclose(
                    q[ids, :n, :n], reference_q[key], atol=1e-12, rtol=0
                ):
                    raise ValueError("Species Q changed")
                row["parts"][key] = dict(
                    features=features[torch.as_tensor(ids, device=device)],
                    target=dd[ids, :n, :n],
                )
        rows.append(row)
    return rows, reference_q, channels


def build_specifications(rows, qs, channels, relative_floor):
    specifications = {}
    for key, ls in channels.items():
        b, w, info = overlap_coordinates(qs[key], ls, relative_floor)
        targets = np.concatenate([r["parts"][key]["target"] for r in rows])
        encoded = w @ targets @ w.T
        origin = scalar_projection(encoded.mean(0), ls)
        scale = max(float(np.mean((encoded - origin) ** 2) ** 0.5), 1e-12)
        specifications[key] = dict(
            channels=ls,
            decode=b.tolist(),
            encode=w.tolist(),
            origin=origin.tolist(),
            scale=scale,
            augmentation_q=qs[key].tolist(),
            diagnostics=info,
        )
    return specifications


def batch(rows, specifications, device):
    result = {}
    for key, s in specifications.items():
        w = np.array(s["encode"])
        origin = np.array(s["origin"])
        target = np.concatenate([r["parts"][key]["target"] for r in rows])
        encoded = (w @ target @ w.T - origin) / s["scale"]
        result[key] = dict(
            features=torch.cat([r["parts"][key]["features"] for r in rows]),
            encoded=torch.tensor(encoded, dtype=torch.float32, device=device),
            target=torch.tensor(target, dtype=torch.float64, device=device),
        )
    return result


def score(model, items):
    rows = {}
    objectives = []
    with torch.no_grad():
        for key, item in items.items():
            head = model.heads[key]
            encoded = head.core(item["features"])
            mse = float((encoded - item["encoded"]).square().mean())
            rows[key] = dict(
                normalized_encoded_mse=mse,
                encoded_rmse=float(mse**0.5 * head.scale),
                onsite_rmse_eV=float(
                    (head.physical(encoded) - item["target"]).square().mean().sqrt()
                ),
            )
            objectives.append(mse)
    return dict(objective=float(np.mean(objectives)), species=rows)


def fit(root, label):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    root = Path(root)
    m = read_json(root / "manifest.json")
    if "trained_root" in m:
        raise ValueError("Evaluation-only run cannot refit models")
    out = root / label
    out.mkdir(exist_ok=False)
    base = load_base_for_repair(m, label, device).eval()
    base.requires_grad_(False)
    training, qs, channels = gather(base, m["entries"]["train"], device)
    specifications = build_specifications(
        training, qs, channels, m["overlap_relative_floor"]
    )
    validation, val_qs, val_channels = gather(base, m["entries"]["validation"], device)
    if channels != val_channels or any(
        not np.allclose(qs[k], val_qs[k], atol=1e-12, rtol=0) for k in qs
    ):
        raise ValueError("Validation PAW contract differs")
    model = ConditionedPAWModel(base, specifications).to(device).eval()
    if "refine_from" in m:
        parent = Path(m["refine_from"])
        check_hash(parent / "manifest.json", m["parent_manifest_sha256"])
        check_hash(parent / label / "model.pt", m["parent_checkpoints"][label])
        previous = torch.load(
            parent / label / "model.pt", map_location="cpu", weights_only=False
        )
        if previous["specifications"] != specifications:
            raise ValueError("Refinement coordinates differ")
        model.heads.load_state_dict(previous["heads"])
    train_items = batch(training, specifications, device)
    val_items = batch(validation, specifications, device)
    optimizer = torch.optim.Adam(model.heads.parameters(), lr=m["learning_rate"])
    use_lbfgs = m.get("optimizer") == "lbfgs"
    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            model.heads.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
    scheduler = (
        None
        if use_lbfgs
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=8, factor=0.5, min_lr=1e-6
        )
    )
    best = score(model, val_items)["objective"]
    best_step = 0
    history = []
    started = time.monotonic()
    best_state = {
        k: v.detach().cpu().clone() for k, v in model.heads.state_dict().items()
    }
    evaluations = 0

    def closure():
        nonlocal evaluations
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        objective = torch.stack(
            [
                (model.heads[k].core(item["features"]) - item["encoded"])
                .square()
                .mean()
                for k, item in train_items.items()
            ]
        ).mean()
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite conditioned loss")
        objective.backward()
        return objective

    for step in range(1, m["max_steps"] + 1):
        if use_lbfgs:
            loss = optimizer.step(closure)
        else:
            loss = closure()
            torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 5.0)
            optimizer.step()
        if step % m["validation_interval"] == 0 or step == m["max_steps"]:
            val = score(model, val_items)
            if scheduler is not None:
                scheduler.step(val["objective"])
            if val["objective"] < best:
                best = val["objective"]
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.heads.state_dict().items()
                }
            history.append(
                dict(
                    step=step,
                    training_objective=float(loss.detach()),
                    validation=val,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            save_json(out / "history.json", history)
            if step % (5 if use_lbfgs else 250) == 0:
                print(
                    label,
                    step,
                    "validation",
                    val,
                    "best",
                    best_step,
                    "seconds",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
        if step >= m["minimum_steps"] and step - best_step >= m["early_stop_patience"]:
            break
    model.heads.load_state_dict(best_state)
    torch.save(
        dict(
            schema="mapho.conditioned_paw.v1",
            specifications=specifications,
            heads=best_state,
            manifest_sha256=sha(root / "manifest.json"),
            label=label,
            best_step=best_step,
        ),
        out / "model.pt",
    )
    save_json(
        out / "training.json",
        dict(
            complete=True,
            best_step=best_step,
            steps=step,
            seed=31,
            train=score(model, train_items),
            validation=score(model, val_items),
            wall_seconds=time.monotonic() - started,
            parameters=sum(p.numel() for p in model.heads.parameters()),
            optimizer="lbfgs" if use_lbfgs else "adam",
            closure_evaluations=evaluations,
            base_checkpoint_sha256=m["models"][label]["checkpoint_sha256"],
            checkpoint_sha256=sha(out / "model.pt"),
            test_labels_used=False,
            field_and_backbone_frozen=True,
        ),
    )
    predictions(root, label, "validation", model=model)


def predictions(root, label, split, model=None):
    torch.set_num_threads(1)
    root = Path(root)
    m = read_json(root / "manifest.json")
    if split == "test" and not read_json(root / "validation_summary.json")["passed"]:
        raise ValueError("Validation must pass before test evaluation")
    model = load_repaired_model(root, label) if model is None else model
    out = root / label / "predictions" / split
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for e in m["entries"][split]:
        # Inference receives no target path. Access below is evaluation only.
        v, d = predict(
            model, {k: e[k] for k in ("baseline", "baseline_sha256")}, "cuda"
        )
        np.savez_compressed(out / f"{e['name']}.npz", potential=v, d=d)
        check_hash(e["target"], e["target_sha256"])
        with h5py.File(e["target"]) as f:
            _, mask = paw_contract(f)
            row = dict(
                name=e["name"],
                field_rmse_eV=float(np.mean((v - f["potential_eV"][:]) ** 2) ** 0.5),
                onsite_rmse_eV=float(
                    np.mean((d[mask] - f["paw/d"][:][mask]) ** 2) ** 0.5
                ),
                prediction_sha256=sha(out / f"{e['name']}.npz"),
            )
        if split == "test":
            activation, mode = label.split("_")
            old = (
                Path(m["training"])
                / activation
                / mode
                / "predictions"
                / f"{e['name']}.npz"
            )
            with np.load(old) as f:
                row["field_unchanged_max_eV"] = float(np.abs(v - f["potential"]).max())
            row["field_unchanged_tolerance_eV"] = m.get(
                "field_reproducibility_tolerance_eV", 1e-5
            )
            if row["field_unchanged_max_eV"] > row["field_unchanged_tolerance_eV"]:
                raise ValueError("Frozen field changed beyond float32 GPU roundoff")
        rows.append(row)
        print(label, split, row, flush=True)
    save_json(root / label / f"{split}_fields.json", dict(rows=rows))


def band_case(root, split, index):
    from .activation_ablation import band_metrics

    root = Path(root)
    m = read_json(root / "manifest.json")
    e = m["entries"][split][index]
    check_hash(m["executable"], m["executable_sha256"])
    check_hash(e["target"], e["target_sha256"])
    folder = root / "bands" / split / e["name"]
    folder.mkdir(parents=True, exist_ok=False)
    points = KS.copy()
    points[:, 2] = 0.0
    spectra = {}
    for label in ["reference", *LABELS]:
        kwargs = {}
        operator = e["target"] if label == "reference" else e["baseline"]
        if label != "reference":
            path = root / label / "predictions" / split / f"{e['name']}.npz"
            row = next(
                r
                for r in read_json(root / label / f"{split}_fields.json")["rows"]
                if r["name"] == e["name"]
            )
            check_hash(path, row["prediction_sha256"])
            with np.load(path) as f:
                kwargs = dict(potential=f["potential"], d=f["d"])
        manifest = prepare_replay(operator, folder / label, kpoints=points, **kwargs)
        manifest["settings"].update(EDIFF=1e-9, NELMIN=32)
        write_incar(folder / label / "INCAR", manifest["settings"])
        manifest["inputs"]["INCAR"] = sha(folder / label / "INCAR")
        save_json(folder / label / "manifest.json", manifest)
        run(folder / label, m["executable"], launcher=LAUNCHER, timeout=1500)
        spectra[label] = read_bands(folder / label)["eigenvalues_eV"]
    # Independent native solve exports ITS operator; replay it to distinguish
    # bridge accuracy from CHGCAR reconstruction vs original SCF differences.
    native = folder / "native"
    native.mkdir()
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        shutil.copyfile(folder / "reference" / name, native / name)
    shutil.copyfile(Path(e["target"]).parent / "CHGCAR", native / "CHGCAR")
    with h5py.File(e["target"]) as f:
        original = json.loads(f.attrs["metadata_json"])["reference_manifest"]
    settings = dict(read_json(folder / "reference/manifest.json")["settings"])
    settings["ICHARG"] = 11
    write_incar(native / "INCAR", settings)
    (native / "MLHAM_EXPORT").touch()
    original.update(
        settings=settings,
        inputs={
            n: sha(native / n)
            for n in ("POSCAR", "POTCAR", "KPOINTS", "INCAR", "CHGCAR")
        },
    )
    save_json(native / "manifest.json", original)
    run(native, m["executable"], launcher=LAUNCHER, timeout=1500)
    export_operator(native, native / "operator.h5")
    prepare_replay(native / "operator.h5", native / "replay", kpoints=points)
    run(native / "replay", m["executable"], launcher=LAUNCHER, timeout=1500)
    spectra["native"] = read_bands(native)["eigenvalues_eV"]
    spectra["native_replay"] = read_bands(native / "replay")["eigenvalues_eV"]
    metrics = {
        label: band_metrics(values, spectra["reference"], m["nocc"], 4)
        for label, values in spectra.items()
    }
    sanity = {
        label: dict(
            all_band_max_eV=float(abs(spectra[label] - spectra["reference"]).max()),
            lowest_eigenvalue_eV=float(spectra[label].min()),
            sampled_gap_eV=float(
                spectra[label][:, m["nocc"]].min()
                - spectra[label][:, m["nocc"] - 1].max()
            ),
        )
        for label in LABELS
    }
    for row in sanity.values():
        row["passed"] = (
            row["all_band_max_eV"] < m["thresholds"]["spectral_all_band_max_eV"]
            and row["sampled_gap_eV"] > m["thresholds"]["minimum_sampled_gap_eV"]
        )
    replay_max = float(abs(spectra["native_replay"] - spectra["native"]).max())
    save_json(
        folder / "report.json",
        dict(
            name=e["name"],
            metrics_vs_scf_export=metrics,
            spectral_sanity=sanity,
            same_operator_replay_max_eV=replay_max,
            same_operator_replay_passed=replay_max
            < m["thresholds"]["same_operator_replay_max_eV"],
            original_scf_vs_chgcar_frontier_max_eV=metrics["native"]["frontier_max_eV"],
            original_reference_gate_passed=metrics["native"]["frontier_max_eV"] < 0.001,
            kpoints=points.tolist(),
            nocc=m["nocc"],
            eigenvalues_eV={k: v.tolist() for k, v in spectra.items()},
        ),
    )


def summarize(root, split):
    root = Path(root)
    m = read_json(root / "manifest.json")
    reports = [
        read_json(root / "bands" / split / e["name"] / "report.json")
        for e in m["entries"][split]
    ]
    result = dict(
        split=split,
        structures=len(reports),
        job=os.environ.get("SLURM_JOB_ID"),
        passed=all(
            r["same_operator_replay_passed"]
            and all(v["passed"] for v in r["spectral_sanity"].values())
            for r in reports
        ),
        thresholds=m["thresholds"],
        models={
            label: dict(
                frontier_mae_eV=float(
                    np.mean(
                        [
                            r["metrics_vs_scf_export"][label]["frontier_mae_eV"]
                            for r in reports
                        ]
                    )
                ),
                all_band_mae_eV=float(
                    np.mean(
                        [
                            r["metrics_vs_scf_export"][label]["all_band_mae_eV"]
                            for r in reports
                        ]
                    )
                ),
                worst_all_band_error_eV=max(
                    r["spectral_sanity"][label]["all_band_max_eV"] for r in reports
                ),
                minimum_sampled_gap_eV=min(
                    r["spectral_sanity"][label]["sampled_gap_eV"] for r in reports
                ),
            )
            for label in LABELS
        },
        max_same_operator_replay_error_eV=max(
            r["same_operator_replay_max_eV"] for r in reports
        ),
        original_scf_vs_chgcar_frontier_max_eV=max(
            r["original_scf_vs_chgcar_frontier_max_eV"] for r in reports
        ),
        original_reference_gate_passed=all(
            r["original_reference_gate_passed"] for r in reports
        ),
        checkpoint_sha256={label: sha(root / label / "model.pt") for label in LABELS},
        caveat="Finite sampled spectra sanity check, not a guarantee excluding ghost states at all geometries/k points; previously examined test structures",
    )
    save_json(root / f"{split}_summary.json", result)
    if not result["passed"]:
        raise RuntimeError(f"{split} spectral validation failed")


def execute(root, phase):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for name, digest in m["source_sha256"].items():
        check_hash(source_path(name), digest)

    def commands(command, labels):
        return [
            (
                f"{command}_{label}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.commands.vasp.train",
                    "paw-repair",
                    command,
                    str(root),
                    label,
                ],
            )
            for label in labels
        ]

    if phase == "validation":
        parallel(root, commands("fit", LABELS), devices)
    else:
        val = read_json(root / "validation_summary.json")
        if not val["passed"]:
            raise ValueError("Validation did not pass")
        for label, digest in val["checkpoint_sha256"].items():
            check_hash(root / label / "model.pt", digest)
        parallel(root, commands("predict-test", LABELS), devices)
    parallel(
        root,
        [
            (
                f"bands_{phase}_{i}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "mapho.commands.vasp.train",
                    "paw-repair",
                    "bands",
                    str(root),
                    phase,
                    str(i),
                ],
            )
            for i in range(len(m["entries"][phase]))
        ],
        devices,
    )
    summarize(root, phase)
