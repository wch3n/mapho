"""CUDA experiment isolating spectral mixing in a fixed exchange-gate ansatz."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from .exchange_ablation import (
    MODES,
    ControlledGate,
    common_state,
    input_statistics,
    prepare_device,
)
from .exchange_train import PreparedExchange
from .field_model import FieldModelConfig, LocalMACEField


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_progress(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def action_mse(s, gate, k):
    error = s.op.apply(s.psi[k, :6], k, gate) - s.target[k, :6]
    return error.abs().square().sum() / 6


def validate(model, samples, ks):
    model.eval()
    scores = []
    with torch.no_grad():
        for s in samples:
            gate = model(s)
            scores.extend(float(action_mse(s, gate, k)) for k in ks)
    return float(np.mean(scores)) ** 0.5 * 1000


def constant_fit(samples, ks):
    aa = 0.0
    ab = 0.0
    with torch.no_grad():
        for s in samples:
            for k in ks:
                a = s.op.apply(s.psi[k, :6], k)
                aa += float(a.abs().square().double().sum())
                ab += float((a.conj() * s.target[k, :6]).real.double().sum())
    beta = float(np.clip(ab / aa, np.exp(-0.2), np.exp(0.2)))
    return {
        "gate": beta**0.5,
        "exchange_multiplier": beta,
        "unconstrained_multiplier": ab / aa,
        "fit": "training structures, k indices 0--5 and band indices 0--5 only",
    }


def train_one(
    args, phase, mode, seed, config, local_state, mean, scale, train, val, protocol_hash
):
    label = f"{phase}_{mode}_seed{seed}"
    out = args.output / label
    out.mkdir(exist_ok=True)
    if (out / "training.json").exists():
        report = json.loads((out / "training.json").read_text())
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("Completed fit protocol mismatch")
        print("reuse completed", label, flush=True)
        return
    torch.manual_seed(seed)
    model = ControlledGate(
        config, mode, local_state, mean.cpu(), scale.cpu(), phase=phase
    ).to(args.device)
    model.local.to(args.local_device)
    initial = cpu_state(model)
    common_hash = hashlib.sha256(
        b"".join(v.numpy().tobytes() for v in common_state(model).values())
    ).hexdigest()
    groups = [{"params": model.correction.parameters(), "lr": 0.001}]
    if phase == "joint":
        groups.append(
            {
                "params": [p for p in model.local.parameters() if p.requires_grad],
                "lr": 0.0001,
            }
        )
    opt = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=0.01,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    ks = (0,) if args.smoke else tuple(range(6))
    check_every = 1 if args.smoke else 6
    best = cpu_state(model)
    best_epoch = 0
    best_score = validate(model, val, ks)
    anchor = best_score
    anchor_epoch = 0
    history = [{"epoch": 0, "validation_rmse_meV": best_score}]
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    reason = "maximum_epochs"
    first_epoch = 1
    previous_seconds = 0.0
    progress = out / "progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_hash:
            raise ValueError("Partial fit protocol mismatch")
        model.load_state_dict(saved["state_dict"])
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["rng"]
        best = saved["best"]
        best_epoch = saved["best_epoch"]
        best_score = saved["best_score"]
        anchor = saved["anchor"]
        anchor_epoch = saved["anchor_epoch"]
        history = saved["history"]
        first_epoch = saved["epoch"] + 1
        previous_seconds = saved["wall_seconds"]
        print("resume", label, "epoch", first_epoch, flush=True)
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    epoch = first_epoch - 1
    already_plateau = (
        epoch >= args.min_epochs and epoch - anchor_epoch >= args.patience_epochs
    )
    if already_plateau:
        reason = "validation_plateau"
    for epoch in range(
        first_epoch, args.max_epochs + 1 if not already_plateau else first_epoch
    ):
        model.train()
        losses = []
        for i in rng.permutation(len(train)):
            s = train[i]
            k = (epoch - 1 + int(i)) % 6
            opt.zero_grad(set_to_none=True)
            gate = model(s)
            loss = action_mse(s, gate, k)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "training_rmse_meV": float(np.mean(losses)) ** 0.5 * 1000,
        }
        if epoch % check_every == 0 or epoch == args.max_epochs:
            score = validate(model, val, ks)
            row["validation_rmse_meV"] = score
            row["learning_rates"] = [g["lr"] for g in opt.param_groups]
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best = cpu_state(model)
            if score < anchor - 0.01:
                anchor = score
                anchor_epoch = epoch
            scheduler.step(score)
            print(
                label,
                "epoch",
                epoch,
                "val_meV",
                round(score, 5),
                "best",
                round(best_score, 5),
                "seconds",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
        history.append(row)
        if epoch % check_every == 0:
            dump(out / "history.json", history)
            save_progress(
                progress,
                {
                    "protocol_sha256": protocol_hash,
                    "epoch": epoch,
                    "state_dict": cpu_state(model),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng.bit_generator.state,
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "anchor": anchor,
                    "anchor_epoch": anchor_epoch,
                    "history": history,
                    "wall_seconds": previous_seconds + time.perf_counter() - start,
                },
            )
        if epoch >= args.min_epochs and epoch - anchor_epoch >= args.patience_epochs:
            reason = "validation_plateau"
            break
    model.load_state_dict(best)
    changed = {
        prefix: sum(
            not torch.equal(v, initial[k])
            for k, v in best.items()
            if k.startswith(prefix)
        )
        for prefix in (
            "local.backbone",
            "local.coefficients",
            "local.offset",
            "correction",
        )
    }
    report = {
        "phase": phase,
        "mode": mode,
        "seed": seed,
        "epochs": epoch,
        "updates": epoch * len(train),
        "best_epoch": best_epoch,
        "best_validation_rmse_meV": best_score,
        "stop_reason": reason,
        "last_meaningful_improvement_epoch": anchor_epoch,
        "plateau_threshold_meV": 0.01,
        "wall_seconds": previous_seconds + time.perf_counter() - start,
        "changed_state_tensors": changed,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "correction_parameters": sum(p.numel() for p in model.correction.parameters()),
        "common_initialization_sha256": common_hash,
        "protocol_sha256": protocol_hash,
        "cuda_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2
        if args.device == "cuda"
        else None,
    }
    torch.save(
        {
            "schema": "mapho.controlled_exchange.v1",
            "state_dict": best,
            "config": asdict(config),
            "metadata": report,
        },
        out / "model.pt",
    )
    dump(out / "history.json", history)
    dump(out / "training.json", report)
    progress.unlink(missing_ok=True)
    print("finished", label, reason, flush=True)


def evaluate(
    args,
    manifest,
    config,
    local_state,
    mean,
    scale,
    reference_local,
    constant,
    protocol_hash,
):
    models = {}
    training = {}
    for phase in args.phases:
        for mode in args.modes:
            for seed in args.seeds:
                label = f"{phase}_{mode}_seed{seed}"
                ck = torch.load(
                    args.output / label / "model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if ck["metadata"]["protocol_sha256"] != protocol_hash:
                    raise ValueError("Checkpoint protocol mismatch")
                m = ControlledGate(
                    config, mode, local_state, mean.cpu(), scale.cpu(), phase=phase
                ).to(args.device)
                m.load_state_dict(ck["state_dict"])
                m.local.to(args.local_device)
                m.eval()
                models[label] = m
                training[label] = ck["metadata"]
    # Check every same-seed, same-phase model truly shared its initialization.
    for phase in args.phases:
        for seed in args.seeds:
            assert (
                len(
                    {
                        training[f"{phase}_{mode}_seed{seed}"][
                            "common_initialization_sha256"
                        ]
                        for mode in args.modes
                    }
                )
                == 1
            )
    rows = []
    gates = []
    with torch.no_grad():
        for name in manifest["splits"]["test"]:
            s = prepare_device(
                PreparedExchange(args.dataset / name, config, target_functional="PBE0"),
                args.device,
                reference_local,
            )
            predicted = {
                "pbe_reference": None,
                "constant": torch.full(s.shape, constant["gate"], device=args.device),
                "pretrained_mace": torch.exp(0.1 * torch.tanh(s.frozen_local)),
            }
            for label, m in models.items():
                predicted[label] = m(s)
            for label, g in predicted.items():
                if g is not None:
                    gates.append(
                        {
                            "sample": name,
                            "model": label,
                            "min": float(g.min()),
                            "max": float(g.max()),
                            "rms_from_one": float((g - 1).square().mean().sqrt()),
                        }
                    )
                for k in range(8):
                    error = s.op.apply(s.psi[k], k, g) - s.target[k]
                    norms = error.flatten(1).norm(dim=1).cpu().numpy()
                    delta = (
                        (s.psi[k].flatten(1).conj() @ error.flatten(1).T)
                        .cpu()
                        .numpy()
                        .astype(np.complex128)
                    )
                    herm = float(np.max(np.abs(delta - delta.conj().T)))
                    if herm > 5e-5 or not np.isfinite(norms).all():
                        raise RuntimeError("Invalid evaluated operator")
                    eig = s.data["eigenvalues"][k]
                    ritz = np.linalg.eigvalsh(
                        np.diag(eig) + 0.5 * (delta + delta.conj().T)
                    )
                    for b in range(8):
                        rows.append(
                            {
                                "sample": name,
                                "model": label,
                                "k": k,
                                "band": b,
                                "action_error_eV": float(norms[b]),
                                "ritz_error_eV": float(ritz[b] - eig[b]),
                                "matrix_hermiticity_error_eV": herm,
                            }
                        )
            print("evaluated", name, flush=True)
    dump(args.output / "evaluation_rows.json", rows)
    dump(args.output / "gates.json", gates)
    groups = {
        "all": lambda r: True,
        "edge_six": lambda r: r["band"] >= 2,
        "upper_valence": lambda r: 2 <= r["band"] < 5,
        "unoccupied": lambda r: r["band"] >= 5,
        "unseen_k": lambda r: r["k"] >= 6,
        "unseen_band": lambda r: r["band"] >= 6,
    }
    metrics = {}
    for label in ["pbe_reference", "constant", "pretrained_mace", *models]:
        rr = [r for r in rows if r["model"] == label]
        metrics[label] = {}
        for group, pred in groups.items():
            gr = [r for r in rr if pred(r)]
            metrics[label][group] = {
                "action_rmse_meV": float(
                    np.mean([r["action_error_eV"] ** 2 for r in gr])
                )
                ** 0.5
                * 1000,
                "ritz_mae_meV": float(np.mean([abs(r["ritz_error_eV"]) for r in gr]))
                * 1000,
            }
    aggregate = {}
    for phase in args.phases:
        aggregate[phase] = {}
        for mode in args.modes:
            aggregate[phase][mode] = {}
            for group in groups:
                values = [
                    metrics[f"{phase}_{mode}_seed{seed}"][group]["action_rmse_meV"]
                    for seed in args.seeds
                ]
                aggregate[phase][mode][group] = {
                    "mean_meV": float(np.mean(values)),
                    "seed_std_meV": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0,
                    "seed_range_meV": [min(values), max(values)],
                }
    paired = {}
    for phase in args.phases:
        if "full" not in args.modes:
            continue
        for other in args.modes:
            if other == "full":
                continue
            diffs = []
            wins = 0
            for seed in args.seeds:
                full = f"{phase}_full_seed{seed}"
                control = f"{phase}_{other}_seed{seed}"
                diffs.append(
                    metrics[full]["all"]["action_rmse_meV"]
                    - metrics[control]["all"]["action_rmse_meV"]
                )
                for name in manifest["splits"]["test"]:
                    a = np.mean(
                        [
                            r["action_error_eV"] ** 2
                            for r in rows
                            if r["model"] == full and r["sample"] == name
                        ]
                    )
                    b = np.mean(
                        [
                            r["action_error_eV"] ** 2
                            for r in rows
                            if r["model"] == control and r["sample"] == name
                        ]
                    )
                    wins += a < b
            paired[f"{phase}_full_minus_{other}"] = {
                "seed_differences_meV": diffs,
                "full_structure_seed_wins": int(wins),
                "structure_seed_count": len(args.seeds)
                * len(manifest["splits"]["test"]),
                "interpretation": "paired descriptive comparisons; structures/seeds are not independent material families",
            }
    summary = {
        "schema": "mapho.spectral_ablation_summary.v1",
        "aggregate": aggregate,
        "metrics": metrics,
        "paired": paired,
        "training": training,
        "constant": constant,
        "protocol_sha256": protocol_hash,
        "scope": "LiF PBE0, existing small-cell distribution; controlled correction comparison, not a test of general long-range transfer. Ritz diagnostics retain exact hybrid local Hamiltonian and eight-state subspace.",
    }
    dump(args.output / "summary.json", summary)
    dump(
        args.output / "completion.json",
        {
            "complete": True,
            "models": len(models),
            "rows": len(rows),
            "all_validation_plateau": all(
                r["stop_reason"] == "validation_plateau" for r in training.values()
            ),
            "summary_sha256": digest(args.output / "summary.json"),
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--local-checkpoint", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 61, 79, 97])
    p.add_argument(
        "--phases", choices=["frozen", "joint"], nargs="+", default=["frozen", "joint"]
    )
    p.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    p.add_argument("--max-epochs", type=int, default=360)
    p.add_argument("--min-epochs", type=int, default=120)
    p.add_argument("--patience-epochs", type=int, default=60)
    p.add_argument("--local-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.dataset = a.dataset.resolve()
    a.output = a.output.resolve()
    a.local_checkpoint = a.local_checkpoint.resolve()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((a.dataset / "manifest.json").read_text())
    if manifest.get("target_functional") != "PBE0":
        raise ValueError("This registered experiment targets LiF PBE0")
    ck = torch.load(a.local_checkpoint, map_location="cpu", weights_only=True)
    if ck["variant"] != "mace" or ck["metadata"]["manifest_sha256"] != digest(
        a.dataset / "manifest.json"
    ):
        raise ValueError("Common MACE checkpoint/dataset mismatch")
    config = FieldModelConfig(**{**ck["config"], "fno_width": 16, "fno_layers": 2})
    local_state = {
        k.removeprefix("local."): v
        for k, v in ck["state_dict"].items()
        if k.startswith("local.")
    }
    reference_local = LocalMACEField(config).to(a.device)
    reference_local.load_state_dict(local_state)
    reference_local.eval()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": str(a.dataset),
        "manifest_sha256": digest(a.dataset / "manifest.json"),
        "local_checkpoint": str(a.local_checkpoint),
        "local_checkpoint_sha256": digest(a.local_checkpoint),
        "config": asdict(config),
        "seeds": a.seeds,
        "phases": a.phases,
        "modes": a.modes,
        "max_epochs": a.max_epochs,
        "min_epochs": a.min_epochs,
        "patience_epochs": a.patience_epochs,
        "plateau_threshold_meV": 0.01,
        "validation_every_epochs": 1 if a.smoke else 6,
        "training": "all six permitted bands at one k per structure per epoch; cycle six permitted k equally",
        "validation": "all six permitted k and bands on two validation structures; no test labels",
        "selection": "minimum validation RMSE, including epoch zero; ReduceLROnPlateau then patience stopping",
        "normalization": "training-only mean and standard deviation of frozen MACE field, cube-root PBE density, PBE Hxc",
        "neighbor_normalization": "retain common pretrained MACE value 16; no architecture-dependent change",
        "optimizer": {"correction_lr": 0.001, "joint_local_lr": 0.0001, "name": "Adam"},
        "source_sha256": {
            f: digest(Path(__file__).parent / f)
            for f in (
                "exchange_ablation.py",
                "exchange_ablation_train.py",
                "exchange.py",
                "exchange_train.py",
                "field_model.py",
            )
        },
        "device": a.device,
        "local_device": a.local_device,
        "smoke": a.smoke,
    }
    if (a.output / "protocol.json").exists():
        if json.loads((a.output / "protocol.json").read_text()) != protocol:
            raise ValueError("Existing output protocol differs")
    else:
        dump(a.output / "protocol.json", protocol)
    protocol_hash = digest(a.output / "protocol.json")
    split = {
        k: v[:n] if a.smoke else v
        for k, v, n in [
            ("train", manifest["splits"]["train"], 2),
            ("validation", manifest["splits"]["validation"], 1),
        ]
    }
    prepared = {}
    for key, names in split.items():
        prepared[key] = []
        for name in names:
            s = prepare_device(
                PreparedExchange(a.dataset / name, config, target_functional="PBE0"),
                a.device,
                reference_local,
            )
            prepared[key].append(s)
            print("prepared", key, name, flush=True)
    train, val = prepared["train"], prepared["validation"]
    mean, scale = input_statistics(train)
    dump(
        a.output / "normalization.json",
        {
            "mean": mean.cpu().tolist(),
            "scale": scale.cpu().tolist(),
            "samples": split["train"],
        },
    )
    constant = constant_fit(train, (0,) if a.smoke else range(6))
    dump(a.output / "constant.json", constant)
    # Round-robin seeds makes intermediate logs cover all architectures fairly.
    for seed in a.seeds:
        for phase in a.phases:
            for mode in a.modes:
                train_one(
                    a,
                    phase,
                    mode,
                    seed,
                    config,
                    local_state,
                    mean,
                    scale,
                    train,
                    val,
                    protocol_hash,
                )
    if not a.smoke:
        del train, val, prepared
        evaluate(
            a,
            manifest,
            config,
            local_state,
            mean,
            scale,
            reference_local,
            constant,
            protocol_hash,
        )
    print("completed spectral ablation", flush=True)


if __name__ == "__main__":
    main()
