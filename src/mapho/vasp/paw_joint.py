"""Joint fine-tuning of conditioned PAW heads, MACE fields and Fourier corrections.

The existing training split and training-only normalization remain fixed.
Validation selects checkpoints; native validation precedes test predictions.
"""

from ..source_snapshot import snapshot_package, source_hashes, source_path
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import torch

from .paw import sha, save_json, prepare_replay, run, read_bands, write_incar
from .paw_repair import load_model as load_repaired, check_hash, read_json
from .paw_conditioning import ConditionedPAWModel
from .train import load_items, paw_contract, predict, cpu_state
from .training_validation import parallel, LAUNCHER, KS

LABELS = ("silu_none", "silu_full")
PREFIXES = (
    "base.local.backbone",
    "base.local.coefficients",
    "base.local.offset",
    "base.correction",
    "heads",
)


def prepare(root, repaired, primitive, epochs=300):
    root, repaired, primitive = map(
        lambda p: Path(p).resolve(), (root, repaired, primitive)
    )
    parent = read_json(repaired / "manifest.json")
    if not read_json(repaired / "validation_summary.json")["passed"]:
        raise ValueError("Repaired models must pass validation")
    if not read_json(primitive / "results.json")["complete"]:
        raise ValueError("Primitive references not certified")
    original = Path(parent["training"]) / "silu"
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    references = {}
    test_eval = root.parent / "vasp_mos2_paw_conditioned_eval_v2"
    for split, source in [("validation", repaired), ("test", test_eval)]:
        summary = read_json(source / f"{split}_summary.json")
        if not summary["passed"]:
            raise ValueError("Certified comparison references required")
        references[split] = {}
        for e in parent["entries"][split]:
            folder = source / "bands" / split / e["name"]
            if not read_json(folder / "reference/run.json")["status"] == "complete":
                raise ValueError("Incomplete comparison reference")
            references[split][e["name"]] = dict(
                directory=str(folder),
                report_sha256=sha(folder / "report.json"),
                reference_eigenval_sha256=sha(folder / "reference/EIGENVAL"),
            )
    save_json(
        root / "manifest.json",
        dict(
            schema="mapho.vasp_conditioned_joint.v1",
            repaired=str(repaired),
            repaired_manifest_sha256=sha(repaired / "manifest.json"),
            parent_checkpoints={s: sha(repaired / s / "model.pt") for s in LABELS},
            training=str(original),
            training_manifest_sha256=sha(original / "manifest.json"),
            entries=parent["entries"],
            references=references,
            primitive=str(primitive),
            primitive_results_sha256=sha(primitive / "results.json"),
            executable=parent["executable"],
            executable_sha256=parent["executable_sha256"],
            nocc=parent["nocc"],
            seed=31,
            epochs=epochs,
            validation_interval=5,
            minimum_epochs=100,
            early_stop_patience=80,
            learning_rates={
                "backbone": 3e-5,
                "field_and_correction": 1e-4,
                "conditioned_heads": 3e-4,
            },
            optimizer="Adam, one full-batch accumulated update per epoch over all 16 training structures",
            objective="field MSE / original training field_scale squared + equal-species normalized encoded PAW MSE",
            selection="Minimum joint validation objective, including the initial repaired checkpoint",
            split_policy="Original splits; no test or primitive labels in fitting, scaling or checkpoint selection",
            normalization="Fixed original field statistics and repaired training-only PAW coordinates",
            thresholds=parent["thresholds"],
            source_sha256=source_hashes(snapshot),
        ),
    )


def initial_model(manifest, label, device):
    parent = Path(manifest["repaired"])
    check_hash(parent / "manifest.json", manifest["repaired_manifest_sha256"])
    check_hash(parent / label / "model.pt", manifest["parent_checkpoints"][label])
    frozen = load_repaired(parent, label, device)
    model = ConditionedPAWModel(frozen.base, frozen.specifications, train_base=True).to(
        device
    )
    model.heads.load_state_dict(frozen.heads.state_dict())
    return model


def data_items(manifest, model, split, device):
    training = Path(manifest["training"])
    check_hash(training / "manifest.json", manifest["training_manifest_sha256"])
    items = load_items(training, split, model.config, device)
    entries = {e["name"]: e for e in manifest["entries"][split]}
    for item in items:
        e = entries[item["name"]]
        for key in ("target", "baseline"):
            check_hash(e[key], e[key + "_sha256"])
        with h5py.File(e["target"]) as t, h5py.File(e["baseline"]) as b:
            contract, mask = paw_contract(t)
            for key in (
                "cell_A",
                "positions_A",
                "atomic_numbers",
                "paw/types",
                "paw/q",
            ):
                if not np.allclose(t[key][:], b[key][:], atol=1e-10, rtol=0):
                    raise ValueError("Geometry/PAW data mismatch")
            numbers = t["atomic_numbers"][:]
            dd = t["paw/d"][:] - b["paw/d"][:]
            item["target_d"] = torch.tensor(dd, dtype=torch.float64, device=device)
            item["d_mask"] = torch.tensor(mask, device=device)
            item["encoded_targets"] = {}
            for z, channels in contract.items():
                key = str(z)
                spec = (
                    model.specifications[key]
                    if key in model.specifications
                    else model.specifications[z]
                )
                if list(channels) != spec["channels"]:
                    raise ValueError("Projector channels differ")
                w = np.array(spec["encode"])
                n = w.shape[0]
                ids = np.flatnonzero(numbers == z)
                if not np.allclose(
                    t["paw/q"][:][ids, :n, :n],
                    spec["augmentation_q"],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("Augmentation Q differs")
                encoded = (w @ dd[ids, :n, :n] @ w.T - np.array(spec["origin"])) / spec[
                    "scale"
                ]
                item["encoded_targets"][key] = torch.tensor(
                    encoded, dtype=torch.float32, device=device
                )
    return items


def losses(model, item):
    field, d, encoded = model(item, return_encoded=True)
    field_mse = (field - item["target"]).square().mean()
    species = {
        k: (x - item["encoded_targets"][k]).square().mean() for k, x in encoded.items()
    }
    paw = torch.stack(list(species.values())).mean()
    objective = field_mse / model.field_scale.square() + paw
    physical = (d - item["target_d"]).square()[item["d_mask"]].mean()
    return objective, field_mse, physical, species


def score(model, items):
    model.eval()
    rows = []
    species = []
    with torch.no_grad():
        for item in items:
            objective, field, physical, parts = losses(model, item)
            rows.append([float(objective), float(field), float(physical)])
            species.append({k: float(v) for k, v in parts.items()})
    avg = np.mean(rows, axis=0)
    return dict(
        objective=float(avg[0]),
        field_rmse_eV=float(avg[1] ** 0.5),
        onsite_rmse_eV=float(avg[2] ** 0.5),
        normalized_paw_mse_by_species={
            k: float(np.mean([s[k] for s in species])) for k in species[0]
        },
    )


def fit(root, label):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(root)
    m = read_json(root / "manifest.json")
    device = "cuda"
    out = root / label
    out.mkdir(exist_ok=False)
    model = initial_model(m, label, device)
    train = data_items(m, model, "train", device)
    validation = data_items(m, model, "validation", device)
    initial = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    parameters = {k: [] for k in m["learning_rates"]}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (
            "backbone"
            if n.startswith("base.local.backbone")
            else "conditioned_heads"
            if n.startswith("heads")
            else "field_and_correction"
        )
        parameters[key].append(p)
    optimizer = torch.optim.Adam(
        [
            dict(params=v, lr=m["learning_rates"][k], name=k)
            for k, v in parameters.items()
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-7
    )
    first = score(model, validation)
    best = first["objective"]
    best_epoch = 0
    best_state = cpu_state(model)
    history = [dict(epoch=0, validation=first)]
    started = time.monotonic()
    rng = np.random.default_rng(m["seed"])
    torch.cuda.reset_peak_memory_stats()
    first_gradients = {}
    for epoch in range(1, m["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_objective = 0.0
        for index in rng.permutation(len(train)):
            objective, *_ = losses(model, train[index])
            if not torch.isfinite(objective):
                raise RuntimeError("Nonfinite joint objective")
            (objective / len(train)).backward()
            training_objective += float(objective.detach()) / len(train)
        if epoch == 1:
            for prefix in PREFIXES:
                grads = [
                    p.grad
                    for n, p in model.named_parameters()
                    if n.startswith(prefix) and p.requires_grad and p.grad is not None
                ]
                first_gradients[prefix] = float(
                    sum(g.double().square().sum() for g in grads).sqrt()
                )
                if not first_gradients[prefix] > 0:
                    raise RuntimeError(f"No joint gradient for {prefix}")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        if epoch % m["validation_interval"] == 0 or epoch == m["epochs"]:
            val = score(model, validation)
            scheduler.step(val["objective"])
            if val["objective"] < best:
                best = val["objective"]
                best_epoch = epoch
                best_state = cpu_state(model)
            history.append(
                dict(
                    epoch=epoch,
                    training_objective=training_objective,
                    validation=val,
                    learning_rates={g["name"]: g["lr"] for g in optimizer.param_groups},
                )
            )
            save_json(out / "history.json", history)
            torch.save(
                dict(
                    schema="mapho.conditioned_paw_joint.v1",
                    state_dict=best_state,
                    manifest_sha256=sha(root / "manifest.json"),
                    label=label,
                    best_epoch=best_epoch,
                ),
                out / "model.pt",
            )
            print(
                label,
                epoch,
                "validation",
                val,
                "best",
                best_epoch,
                "seconds",
                round(time.monotonic() - started, 1),
                flush=True,
            )
        if (
            epoch >= m["minimum_epochs"]
            and epoch - best_epoch >= m["early_stop_patience"]
        ):
            break
    model.load_state_dict(best_state)
    changes = {}
    for prefix in PREFIXES:
        deltas = [
            (p.detach().cpu() - initial[n]).double()
            for n, p in model.named_parameters()
            if n in initial and n.startswith(prefix)
        ]
        changes[prefix] = dict(
            changed_tensors=sum(bool(torch.count_nonzero(d)) for d in deltas),
            l2=float(sum(d.square().sum() for d in deltas).sqrt()),
        )
    save_json(
        out / "training.json",
        dict(
            complete=True,
            label=label,
            epochs=epoch,
            updates=epoch,
            best_epoch=best_epoch,
            structures_per_update=len(train),
            training=score(model, train),
            validation=score(model, validation),
            initial_validation=first,
            first_update_gradient_norms=first_gradients,
            parameter_changes=changes,
            all_active_groups_changed=all(v["l2"] > 0 for v in changes.values()),
            checkpoint_sha256=sha(out / "model.pt"),
            seed=m["seed"],
            wall_seconds=time.monotonic() - started,
            cuda_peak_MiB=torch.cuda.max_memory_allocated() / 1024**2,
            fitted_with_test_or_primitive_labels=False,
            trainable="MACE backbone, field decoder and offset, correction network, conditioned PAW heads",
        ),
    )


def load_model(root, label, device="cuda"):
    root = Path(root)
    m = read_json(root / "manifest.json")
    check_hash(
        root / label / "model.pt",
        read_json(root / label / "training.json")["checkpoint_sha256"],
    )
    checkpoint = torch.load(
        root / label / "model.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint["manifest_sha256"] != sha(root / "manifest.json")
        or checkpoint["label"] != label
    ):
        raise ValueError("Joint checkpoint contract changed")
    model = initial_model(m, label, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def predictions(root, label, split):
    torch.set_num_threads(1)
    root = Path(root)
    m = read_json(root / "manifest.json")
    if (
        split != "validation"
        and not read_json(root / "validation_summary.json")["passed"]
    ):
        raise ValueError("Native validation must pass before test")
    model = load_model(root, label)
    out = root / label / "predictions" / split
    out.mkdir(parents=True, exist_ok=False)
    if split == "primitive":
        baseline = Path(m["primitive"]) / "mos2/references/atomic/operator.h5"
        entries = [
            dict(
                name="primitive", baseline=str(baseline), baseline_sha256=sha(baseline)
            )
        ]
    else:
        entries = m["entries"][split]
    rows = []
    for e in entries:
        v, d = predict(
            model, {k: e[k] for k in ("baseline", "baseline_sha256")}, "cuda"
        )
        path = out / f"{e['name']}.npz"
        np.savez_compressed(path, potential=v, d=d)
        row = dict(name=e["name"], prediction_sha256=sha(path))
        if split != "primitive":
            check_hash(e["target"], e["target_sha256"])
            with h5py.File(e["target"]) as f:
                _, mask = paw_contract(f)
                row.update(
                    field_rmse_eV=float(
                        np.mean((v - f["potential_eV"][:]) ** 2) ** 0.5
                    ),
                    onsite_rmse_eV=float(
                        np.mean((d[mask] - f["paw/d"][:][mask]) ** 2) ** 0.5
                    ),
                )
        rows.append(row)
    save_json(root / label / f"{split}_fields.json", dict(rows=rows))


def bands(root, split, index, label):
    root = Path(root)
    m = read_json(root / "manifest.json")
    check_hash(m["executable"], m["executable_sha256"])
    if split == "primitive":
        primitive = Path(m["primitive"])
        check_hash(primitive / "results.json", m["primitive_results_sha256"])
        name = "primitive"
        baseline = primitive / "mos2/references/atomic/operator.h5"
        points = read_json(primitive / "manifest.json")["materials"]["mos2"]["path"][
            "kpoints"
        ]
    else:
        e = m["entries"][split][index]
        name = e["name"]
        baseline = Path(e["baseline"])
        check_hash(baseline, e["baseline_sha256"])
        points = KS.copy()
        points[:, 2] = 0.0
    path = root / label / "predictions" / split / f"{name}.npz"
    row = next(
        r
        for r in read_json(root / label / f"{split}_fields.json")["rows"]
        if r["name"] == name
    )
    check_hash(path, row["prediction_sha256"])
    folder = root / "bands" / split / name / label
    with np.load(path) as f:
        manifest = prepare_replay(
            baseline, folder, potential=f["potential"], d=f["d"], kpoints=points
        )
    manifest["settings"].update(EDIFF=1e-9, NELMIN=32, ISYM=-1)
    write_incar(folder / "INCAR", manifest["settings"])
    manifest["inputs"]["INCAR"] = sha(folder / "INCAR")
    save_json(folder / "manifest.json", manifest)
    run(folder, m["executable"], launcher=LAUNCHER, timeout=1800)


def summarize(root, split):
    from .activation_ablation import band_metrics

    root = Path(root)
    m = read_json(root / "manifest.json")
    reports = []
    for e in m["entries"][split]:
        name = e["name"]
        source = m["references"][split][name]
        folder = Path(source["directory"])
        check_hash(folder / "report.json", source["report_sha256"])
        check_hash(folder / "reference/EIGENVAL", source["reference_eigenval_sha256"])
        old = read_json(folder / "report.json")
        ref = np.asarray(old["eigenvalues_eV"]["reference"])
        rows = {}
        for label in LABELS:
            data = read_bands(root / "bands" / split / name / label)
            if not np.allclose(data["kpoints"], old["kpoints"], atol=1e-7, rtol=0):
                raise ValueError("Comparison k points differ")
            values = data["eigenvalues_eV"]
            gap = float(values[:, m["nocc"]].min() - values[:, m["nocc"] - 1].max())
            metric = band_metrics(values, ref, m["nocc"], 4)
            maximum = float(abs(values - ref).max())
            rows[label] = dict(
                metrics=metric,
                before_metrics=old["metrics_vs_scf_export"][label],
                sampled_gap_eV=gap,
                all_band_max_eV=maximum,
                passed=maximum < m["thresholds"]["spectral_all_band_max_eV"]
                and gap > 0,
                eigenvalues_eV=values.tolist(),
            )
        reports.append(dict(name=name, models=rows))
    summary = dict(
        split=split,
        passed=all(s["passed"] for r in reports for s in r["models"].values()),
        reports=reports,
        checkpoint_sha256={label: sha(root / label / "model.pt") for label in LABELS},
        models={
            label: dict(
                frontier_mae_eV=float(
                    np.mean(
                        [
                            r["models"][label]["metrics"]["frontier_mae_eV"]
                            for r in reports
                        ]
                    )
                ),
                before_frontier_mae_eV=float(
                    np.mean(
                        [
                            r["models"][label]["before_metrics"]["frontier_mae_eV"]
                            for r in reports
                        ]
                    )
                ),
                worst_all_band_error_eV=max(
                    r["models"][label]["all_band_max_eV"] for r in reports
                ),
                minimum_sampled_gap_eV=min(
                    r["models"][label]["sampled_gap_eV"] for r in reports
                ),
            )
            for label in LABELS
        },
        caveat="Previously examined test structures; no test fitting or selection; finite spectral sanity check",
    )
    save_json(root / f"{split}_summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Joint native spectral sanity check failed")
    return summary


def primitive_results(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    source = Path(m["primitive"])
    check_hash(source / "results.json", m["primitive_results_sha256"])
    data = copy.deepcopy(read_json(source / "results.json"))
    r = data["materials"]["mos2"]
    ref = np.asarray(r["eigenvalues_eV"]["native_high"])
    nocc = r["specification"]["nocc"]
    r["before_fields"] = copy.deepcopy(r["fields"])
    r["fields"] = {}
    r["specification"].update(
        training=str(root),
        joint_finetuning=True,
        primitive_labels_used_for_training=False,
        parent_band_result=str(source / "results.json"),
        checkpoints={
            mode: sha(root / ("silu_" + mode) / "model.pt") for mode in ("none", "full")
        },
    )
    for mode in ("none", "full"):
        label = "silu_" + mode
        folder = root / "bands/primitive/primitive" / label
        with (
            np.load(root / label / "predictions/primitive/primitive.npz") as f,
            h5py.File(source / "mos2/references/scf_high/operator.h5") as target,
        ):
            _, mask = paw_contract(target)
            r["fields"][mode] = dict(
                potential_rmse_eV=float(
                    np.mean((f["potential"] - target["potential_eV"][:]) ** 2) ** 0.5
                ),
                onsite_rmse_eV=float(
                    np.mean((f["d"][mask] - target["paw/d"][:][mask]) ** 2) ** 0.5
                ),
            )
        values = read_bands(folder)
        if not np.allclose(
            values["kpoints"], r["specification"]["path"]["kpoints"], atol=1e-7, rtol=0
        ):
            raise ValueError("Primitive path changed")
        e = values["eigenvalues_eV"]
        d = e - ref
        window = slice(nocc - 2, nocc + 2)
        vi = int(np.argmax(e[:, nocc - 1]))
        ci = int(np.argmin(e[:, nocc]))
        r["eigenvalues_eV"]["before_" + mode] = r["eigenvalues_eV"][mode]
        r["metrics"]["before_" + mode] = r["metrics"][mode]
        r["eigenvalues_eV"][mode] = e.tolist()
        r["metrics"][mode] = dict(
            all_band_mae_eV=float(abs(d).mean()),
            all_band_max_eV=float(abs(d).max()),
            frontier_mae_eV=float(abs(d[:, window]).mean()),
            frontier_max_eV=float(abs(d[:, window]).max()),
            vbm_eV=float(e[vi, nocc - 1]),
            cbm_eV=float(e[ci, nocc]),
            sampled_gap_eV=float(e[ci, nocc] - e[vi, nocc - 1]),
            vbm_index=vi,
            cbm_index=ci,
        )
        r["source_sha256"][str(folder / "EIGENVAL")] = sha(folder / "EIGENVAL")
    previous = (
        read_json(root / "results.json") if (root / "results.json").exists() else {}
    )
    job = os.environ.get("SLURM_JOB_ID") or previous.get("job")
    data.update(
        job=job,
        joint_finetuning=True,
        scope="LiF unchanged joint model; MoS2 jointly fine-tuned conditioned model",
        joint_manifest_sha256=sha(root / "manifest.json"),
    )
    save_json(root / "results.json", data)
    return data


def execute(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for name, digest in m["source_sha256"].items():
        check_hash(source_path(name), digest)

    def command(stage, *args):
        return (
            "_".join((stage, *map(str, args))),
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                stage,
                str(root),
                *map(str, args),
            ],
        )

    parallel(root, [command("fit", s) for s in LABELS], devices)
    parallel(root, [command("predict", s, "validation") for s in LABELS], devices)
    parallel(
        root,
        [
            command("bands", "validation", i, s)
            for i in range(len(m["entries"]["validation"]))
            for s in LABELS
        ],
        devices,
    )
    summarize(root, "validation")
    parallel(
        root,
        [
            command("predict", s, split)
            for split in ("test", "primitive")
            for s in LABELS
        ],
        devices,
    )
    parallel(
        root,
        [
            command("bands", "test", i, s)
            for i in range(len(m["entries"]["test"]))
            for s in LABELS
        ]
        + [command("bands", "primitive", 0, s) for s in LABELS],
        devices,
    )
    summarize(root, "test")
    primitive_results(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("repaired")
    q.add_argument("primitive")
    q.add_argument("--epochs", type=int, default=300)
    for name in ("execute", "primitive-results"):
        q = sub.add_parser(name)
        q.add_argument("root")
    for name in ("fit", "predict"):
        q = sub.add_parser(name)
        q.add_argument("root")
        q.add_argument("label", choices=LABELS)
        if name == "predict":
            q.add_argument("split", choices=["validation", "test", "primitive"])
    q = sub.add_parser("bands")
    q.add_argument("root")
    q.add_argument("split")
    q.add_argument("index", type=int)
    q.add_argument("label", choices=LABELS)
    q = sub.add_parser("summarize")
    q.add_argument("root")
    q.add_argument("split")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.root, a.repaired, a.primitive, a.epochs)
    elif a.command == "fit":
        fit(a.root, a.label)
    elif a.command == "predict":
        predictions(a.root, a.label, a.split)
    elif a.command == "bands":
        bands(a.root, a.split, a.index, a.label)
    elif a.command == "summarize":
        summarize(a.root, a.split)
    elif a.command == "primitive-results":
        primitive_results(a.root)
    else:
        execute(a.root)
