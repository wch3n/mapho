"""Train and fine-tune MACE and FNO models of DFT potential fields."""

from .schema import canonical_schema

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .fields import FieldSample, load_operator_reference
from .field_model import (
    FieldInputs,
    FieldModelConfig,
    LocalMACEField,
    ResidualFNO,
    mace_graph,
    resize_field,
    predict_fields,
    save_field_models,
    load_field_models,
)
from .probes import PeriodicProbeGrid


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def contract(sample):
    m = sample.metadata
    return {
        "functional": m["functional"],
        "nspin": m["nspin"],
        "potential_gauge": m["potential_gauge"],
        "reference_definition": m["reference_definition"],
        "pseudopotentials": {s: v["sha256"] for s, v in m["pseudopotentials"].items()},
        "qe_config": m["qe_config"],
    }


class FieldFiles:
    """Index paths only; each access loads exactly one validated structure."""

    def __init__(self, paths):
        self.paths = tuple(map(Path, paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return path, FieldSample.load(path)


def load_dataset(directory):
    root = Path(directory).resolve()
    if not (root / "complete.json").is_file():
        raise ValueError("Dataset generation is incomplete")
    manifest = json.loads((root / "manifest.json").read_text())
    if canonical_schema(manifest.get("schema")) != "mapho.field_dataset.v1":
        raise ValueError("Unsupported dataset manifest")
    if set(manifest["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    datasets, seen, expected = {}, set(), None
    for name, paths in manifest["splits"].items():
        if not paths:
            raise ValueError("Every split must be nonempty")
        split_paths = []
        for relative in paths:
            path = root / relative
            sample = FieldSample.load(path)
            key = hashlib.sha256(
                sample.numbers.tobytes()
                + sample.positions.tobytes()
                + sample.cell.tobytes()
            ).hexdigest()
            if key in seen:
                raise ValueError("Repeated structure across dataset entries")
            seen.add(key)
            if expected is None:
                expected = contract(sample)
            if expected != contract(sample):
                raise ValueError(
                    "Inconsistent pseudopotential, gauge, reference, or QE settings"
                )
            split_paths.append(path)
        datasets[name] = FieldFiles(split_paths)
    return datasets, manifest, expected


class PreparedFields:
    """One structure's graph, periodic index and coarse targets, without caching."""

    def __init__(self, entries, config):
        self.entries, self.config = entries, config

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, sample = self.entries[index]
        config = self.config
        shape = (config.grid_size,) * 3
        inputs = FieldInputs.from_sample(sample)
        return {
            "path": path,
            "graph": mace_graph(inputs, config),
            "probes": PeriodicProbeGrid(inputs, shape, config),
            "cell": torch.tensor(inputs.cell, dtype=torch.float32),
            "reference": torch.tensor(
                np.stack(
                    [
                        resize_field(inputs.reference_density, shape),
                        resize_field(inputs.reference_hxc, shape),
                    ]
                ),
                dtype=torch.float32,
            ),
            # Train the absolute residual; never insert label means at inference.
            "target": torch.tensor(
                resize_field(sample.hxc - inputs.reference_hxc, shape),
                dtype=torch.float32,
            ),
        }


def _prepare(entries, config):
    return PreparedFields(entries, config)


def _local_predict(model, item):
    return model(item["graph"], item["probes"]).reshape(item["target"].shape)


class ResidualFiles:
    """Stream compact residual-stage tensors; no atomic graphs or probe caches."""

    def __init__(self, paths):
        self.paths = tuple(map(Path, paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return {
                key: torch.from_numpy(data[key])
                for key in ("channels", "target", "cell")
            }


@torch.no_grad()
def cache_residuals(local, entries, directory):
    """Freeze local predictions to disk, returning streaming training statistics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    local.eval()
    paths = []
    input_squares = torch.zeros(3, dtype=torch.float64)
    target_square = 0.0
    for index, item in enumerate(entries):
        prediction = _local_predict(local, item)
        channels = torch.cat([prediction[None], item["reference"]])
        target = item["target"] - prediction
        path = directory / f"sample_{index:06d}.npz"
        with path.open("xb") as stream:
            np.savez(
                stream,
                channels=channels.numpy(),
                target=target.numpy(),
                cell=item["cell"].numpy(),
            )
        paths.append(path)
        input_squares += channels.square().mean((1, 2, 3)).double()
        target_square += float(target.square().mean())
    return (
        ResidualFiles(paths),
        (input_squares / len(paths)).sqrt().float(),
        (target_square / len(paths)) ** 0.5,
    )


class CoupledField(torch.nn.Module):
    """Combined residual with gradients through both uses of the local field."""

    def __init__(self, local, fno):
        super().__init__()
        self.local, self.fno = local, fno

    def forward(self, item):
        local = _local_predict(self.local, item)
        channels = torch.cat([local[None], item["reference"]])
        return local + self.fno(channels, item["cell"])


def _fit(
    model,
    train,
    validation,
    predict,
    epochs,
    lr,
    scale,
    seed,
    label,
    parameter_groups=None,
):
    parameters = (
        [p for p in model.parameters() if p.requires_grad]
        if parameter_groups is None
        else parameter_groups
    )
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # Apply the same relative decay to each group, preserving the MACE/FNO ratio.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.03 + 0.97 * (1 + np.cos(np.pi * step / epochs)) / 2
    )
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = float("inf"), None, -1
    history = []
    for epoch in range(epochs + 1):
        model.eval()
        with torch.no_grad():
            val = np.mean(
                [
                    float((predict(model, d) - d["target"]).square().mean())
                    for d in validation
                ]
            )
        if val < best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        row = {"epoch": epoch, "validation_rmse_eV": float(np.sqrt(val))}
        if epoch == epochs:
            history.append(row)
            break
        model.train()
        losses = []
        for index in rng.permutation(len(train)):
            d = train[index]
            optimizer.zero_grad(set_to_none=True)
            loss = (predict(model, d) - d["target"]).square().mean() / scale**2
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            losses.append(float(loss.detach()) * scale**2)
        scheduler.step()
        row["train_rmse_eV"] = float(np.sqrt(np.mean(losses)))
        history.append(row)
        if epoch % 10 == 0:
            print(f"{label} epoch {epoch}: val RMSE {np.sqrt(val):.6f} eV", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": best_epoch,
        "best_validation_rmse_eV": float(np.sqrt(best)),
        "history": history,
    }


def train(
    dataset,
    output,
    config=None,
    local_epochs=160,
    fno_epochs=100,
    seed=31,
    probe_chunk_size=None,
    probe_edge_chunk_size=None,
    spectral_chunk_size=None,
):
    if min(local_epochs, fno_epochs) < 1:
        raise ValueError("Training epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.perf_counter()
    c = config or FieldModelConfig()
    _set_chunks(c, probe_chunk_size, probe_edge_chunk_size, spectral_chunk_size)
    c.validate()
    datasets, manifest, expected = load_dataset(dataset)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "split.json", manifest)
    local, fno = LocalMACEField(c), ResidualFNO(c)
    # No held-out fields enter optimization, normalization, or model selection.
    train_data = _prepare(datasets["train"], c)
    val_data = _prepare(datasets["validation"], c)
    scale = (
        sum(float(d["target"].square().mean()) for d in train_data) / len(train_data)
    ) ** 0.5
    scale = max(scale, 1e-8)
    local_history = _fit(
        local,
        train_data,
        val_data,
        _local_predict,
        local_epochs,
        0.005,
        scale,
        seed,
        "local",
    )
    local.requires_grad_(False)
    local.zero_grad(set_to_none=True)
    residual_train, input_scale, residual_scale = cache_residuals(
        local, train_data, out / "fno_cache" / "train"
    )
    residual_val, _, _ = cache_residuals(
        local, val_data, out / "fno_cache" / "validation"
    )
    # FNO training retains neither local graphs nor probe geometry/bases.
    del train_data, val_data
    fno.input_scale.copy_(input_scale.clamp_min(1e-8))
    fno.output_scale.fill_(max(residual_scale, 1e-8))
    fno_history = _fit(
        fno,
        residual_train,
        residual_val,
        lambda m, d: m(d["channels"], d["cell"]),
        fno_epochs,
        0.003,
        float(fno.output_scale),
        seed + 1,
        "fno",
    )
    metadata = {
        "seed": seed,
        "contract": expected,
        "config": asdict(c),
        "local_epochs": local_epochs,
        "fno_epochs": fno_epochs,
        "local_parameters": sum(p.numel() for p in local.parameters()),
        "fno_parameters": sum(p.numel() for p in fno.parameters()),
        "dataset_manifest_sha256": hashlib.sha256(
            (Path(dataset) / "manifest.json").read_bytes()
        ).hexdigest(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("torch", "e3nn", "mace-torch", "numpy", "scipy", "ase")
        },
        "wall_seconds": time.perf_counter() - started,
    }
    save_field_models(out / "model.pt", local, fno, metadata)
    write_json(out / "history.json", {"local": local_history, "fno": fno_history})
    write_json(out / "runtime.json", metadata)
    print(f"Saved {out / 'model.pt'}", flush=True)
    return local, fno


def _parameter_changes(model, initial):
    squares, maximum, changed = 0.0, 0.0, 0
    for name, parameter in model.named_parameters():
        if not parameter.numel():
            continue
        delta = parameter.detach().double() - initial[name].double()
        squares += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(torch.count_nonzero(delta) > 0)
    return {"l2": squares**0.5, "max_abs": maximum, "changed_tensors": changed}


def finetune(
    dataset,
    checkpoint,
    output,
    mode="joint",
    epochs=50,
    local_lr=1e-4,
    fno_lr=3e-4,
    seed=47,
    probe_chunk_size=None,
    probe_edge_chunk_size=None,
    spectral_chunk_size=None,
):
    """Continue a staged checkpoint, with a frozen-local control or joint fitting.

    Both modes retain checkpoint normalization and start fresh Adam optimizers.
    Only validation potential error selects weights; epoch zero is eligible.
    """
    if mode not in ("joint", "fno"):
        raise ValueError("Fine-tuning mode must be joint or fno")
    if int(epochs) != epochs or epochs < 1:
        raise ValueError("Fine-tuning epochs must be a positive integer")
    if not all(np.isfinite(lr) and lr > 0 for lr in (local_lr, fno_lr)):
        raise ValueError("Fine-tuning learning rates must be finite and positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.perf_counter()
    datasets, manifest, expected = load_dataset(dataset)
    local, fno, parent_metadata = load_field_models(checkpoint)
    _runtime_chunks(
        local, fno, probe_chunk_size, probe_edge_chunk_size, spectral_chunk_size
    )
    manifest_hash = hashlib.sha256(
        (Path(dataset) / "manifest.json").read_bytes()
    ).hexdigest()
    if expected != parent_metadata["contract"]:
        raise ValueError("Checkpoint and dataset contracts differ")
    if manifest_hash != parent_metadata["dataset_manifest_sha256"]:
        raise ValueError("Fine-tuning split differs from the training manifest")
    scale = float(fno.output_scale)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not torch.isfinite(fno.input_scale).all()
        or (fno.input_scale <= 0).any()
    ):
        raise ValueError("Invalid checkpoint normalization")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "split.json", manifest)
    local.requires_grad_(mode == "joint")
    # Unfreezing the field model must not enable unused MACE energy heads.
    local.backbone.readouts.requires_grad_(False)
    fno.requires_grad_(True)
    components = {
        "local": local,
        "mace_backbone": local.backbone,
        "field_coefficients": local.coefficients,
        "field_offset": local.offset,
        "unused_energy_readouts": local.backbone.readouts,
        "fno": fno,
    }
    initial = {
        key: {name: p.detach().clone() for name, p in module.named_parameters()}
        for key, module in components.items()
    }
    train_data = _prepare(datasets["train"], local.config)
    val_data = _prepare(datasets["validation"], local.config)
    if mode == "joint":
        model = CoupledField(local, fno)
        groups = [
            {
                "params": [p for p in local.parameters() if p.requires_grad],
                "lr": local_lr,
            },
            {"params": list(fno.parameters()), "lr": fno_lr},
        ]
        history = _fit(
            model,
            train_data,
            val_data,
            lambda m, d: m(d),
            epochs,
            fno_lr,
            scale,
            seed,
            "joint",
            parameter_groups=groups,
        )
    else:
        local.eval()
        residual_train, _, _ = cache_residuals(
            local, train_data, out / "fno_cache" / "train"
        )
        residual_val, _, _ = cache_residuals(
            local, val_data, out / "fno_cache" / "validation"
        )
        del train_data, val_data
        history = _fit(
            fno,
            residual_train,
            residual_val,
            lambda m, d: m(d["channels"], d["cell"]),
            epochs,
            fno_lr,
            scale,
            seed,
            "fno_continuation",
        )
    changes = {
        key: _parameter_changes(module, initial[key])
        for key, module in components.items()
    }
    metadata = copy.deepcopy(parent_metadata)
    metadata.update(
        config=asdict(local.config),
        wall_seconds=time.perf_counter() - started,
        parameter_changes=changes,
    )
    metadata["fine_tuning"] = {
        "mode": mode,
        "epochs": epochs,
        "seed": seed,
        "local_lr": local_lr if mode == "joint" else 0.0,
        "fno_lr": fno_lr,
        "optimizer": "fresh Adam",
        "scheduler": "cosine multiplier, final factor 0.03",
        "normalization": "unchanged from parent checkpoint",
        "selection": "minimum validation potential MSE, including epoch zero",
        "best_epoch": history["best_epoch"],
        "optimizer_steps": epochs * len(datasets["train"]),
        "parent_checkpoint": str(Path(checkpoint).resolve()),
        "parent_checkpoint_sha256": hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
    }
    save_field_models(out / "model.pt", local, fno, metadata)
    write_json(out / "history.json", {"finetune": history})
    write_json(out / "runtime.json", metadata)
    print(f"Saved {out / 'model.pt'}", flush=True)
    return local, fno


def potential_metrics(prediction, sample):
    difference = prediction - sample.hxc
    return {
        "potential_mae_eV": float(np.abs(difference).mean()),
        "potential_rmse_eV": float(np.sqrt(np.mean(difference**2))),
        "potential_mean_error_eV": float(difference.mean()),
        "potential_centered_rmse_eV": float(np.std(difference)),
    }


def electronic_metrics(solution, sample):
    difference = solution["eigenvalues_eV"] - sample.eigenvalues
    weight = sample.kweights[:, None] / difference.shape[1]
    shift = float((difference * weight).sum())
    occupied = sample.occupations > 1
    if not np.all(occupied.sum(1) == occupied.sum(1)[0]):
        raise ValueError("Expected an insulating fixed-occupation reference")
    gap = lambda e: float(e[~occupied].min() - e[occupied].max())
    predicted_gap, reference_gap = (
        gap(solution["eigenvalues_eV"]),
        gap(sample.eigenvalues),
    )
    return {
        "band_mae_eV": float((np.abs(difference) * weight).sum()),
        "band_max_error_eV": float(np.abs(difference).max()),
        "band_mean_error_eV": shift,
        "band_aligned_mae_eV": float((np.abs(difference - shift) * weight).sum()),
        "sampled_gap_eV": predicted_gap,
        "reference_sampled_gap_eV": reference_gap,
        "sampled_gap_abs_error_eV": abs(predicted_gap - reference_gap),
        "density_relative_l1": float(
            np.abs(solution["density"] - sample.fields["density"]).sum()
            * sample.dv
            / sample.metadata["nelect"]
        ),
    }


def _set_chunks(
    config, probe_chunk_size=None, probe_edge_chunk_size=None, spectral_chunk_size=None
):
    for key, value in (
        ("probe_chunk_size", probe_chunk_size),
        ("probe_edge_chunk_size", probe_edge_chunk_size),
        ("spectral_chunk_size", spectral_chunk_size),
    ):
        if value is not None:
            setattr(config, key, value)
    config.validate()


def _runtime_chunks(
    local,
    fno,
    probe_chunk_size=None,
    probe_edge_chunk_size=None,
    spectral_chunk_size=None,
):
    _set_chunks(
        local.config, probe_chunk_size, probe_edge_chunk_size, spectral_chunk_size
    )
    for layer in fno.layers:
        layer.chunk_size = local.config.spectral_chunk_size


def evaluate(
    dataset,
    checkpoint,
    output,
    qe_python=None,
    probe_chunk_size=None,
    probe_edge_chunk_size=None,
    spectral_chunk_size=None,
    models=None,
):
    models = ("atomic", "local", "local_fno") if models is None else tuple(models)
    if (
        not models
        or len(set(models)) != len(models)
        or not set(models) <= {"atomic", "local", "local_fno"}
    ):
        raise ValueError("Choose distinct models from atomic, local, local_fno")
    torch.set_num_threads(1)
    datasets, manifest, expected = load_dataset(dataset)
    local, fno, metadata = load_field_models(checkpoint)
    _runtime_chunks(
        local, fno, probe_chunk_size, probe_edge_chunk_size, spectral_chunk_size
    )
    if expected != metadata["contract"]:
        raise ValueError("Checkpoint and dataset contracts differ")
    manifest_hash = hashlib.sha256(
        (Path(dataset) / "manifest.json").read_bytes()
    ).hexdigest()
    if manifest_hash != metadata["dataset_manifest_sha256"]:
        raise ValueError("Test split differs from the training manifest")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, (path, sample) in enumerate(datasets["test"]):
        inputs = FieldInputs.from_sample(sample)
        predictions = {
            name: (
                inputs.reference_hxc
                if name == "atomic"
                else predict_fields(local, fno if name == "local_fno" else None, inputs)
            )
            for name in models
        }
        record = {
            "sample": str(path.relative_to(Path(dataset).resolve())),
            "models": {},
        }
        for name, predicted in predictions.items():
            target = out / f"test_{index:03d}_{name}.npy"
            np.save(target, predicted, allow_pickle=False)
            metrics = potential_metrics(predicted, sample)
            if qe_python:
                run_dir = out / f"test_{index:03d}_{name}"
                with (out / f"test_{index:03d}_{name}.log").open("x") as log:
                    # Preserve the venv executable symlink; resolving it selects
                    # the base interpreter and loses the isolated QE packages.
                    subprocess.run(
                        [
                            str(Path(qe_python).expanduser().absolute()),
                            "-m",
                            "mapho.qe",
                            "solve",
                            "--reference",
                            str(path),
                            "--hxc",
                            str(target),
                            "--output",
                            str(run_dir),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                with np.load(run_dir / "solution.npz", allow_pickle=False) as solution:
                    metrics.update(electronic_metrics(solution, sample))
            record["models"][name] = metrics
        rows.append(record)
        print(
            f"Evaluated held-out structure {index + 1}/{len(datasets['test'])}",
            flush=True,
        )
    mean = {
        name: {
            key: float(np.mean([r["models"][name][key] for r in rows]))
            for key in rows[0]["models"][name]
        }
        for name in predictions
    }
    report = {
        "schema": "mapho.field_evaluation.v1",
        "aggregation": "equal weight per held-out structure",
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "count": len(rows),
        "qe_solved": bool(qe_python),
        "mean": mean,
        "samples": rows,
    }
    write_json(out / "metrics.json", report)
    print(json.dumps(mean, indent=2), flush=True)
    return report


def predict(
    reference,
    checkpoint,
    output,
    local_only=False,
    probe_chunk_size=None,
    probe_edge_chunk_size=None,
    spectral_chunk_size=None,
):
    """Predict from an SCF-free atomic reference, retaining the native QE gauge."""
    torch.set_num_threads(1)
    sample = load_operator_reference(reference)
    local, fno, metadata = load_field_models(checkpoint)
    _runtime_chunks(
        local, fno, probe_chunk_size, probe_edge_chunk_size, spectral_chunk_size
    )
    if contract(sample) != metadata["contract"]:
        raise ValueError("Inference reference differs from the checkpoint contract")
    prediction = predict_fields(
        local, None if local_only else fno, FieldInputs.from_sample(sample)
    )
    with Path(output).open("xb") as stream:
        np.save(stream, prediction, allow_pickle=False)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--local-epochs", type=int, default=160)
    p.add_argument("--fno-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=31)
    p = commands.add_parser("finetune")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("joint", "fno"), default="joint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--local-lr", type=float, default=1e-4)
    p.add_argument("--fno-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=47)
    p = commands.add_parser("evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qe-python")
    p.add_argument("--models", choices=("atomic", "local", "local_fno"), nargs="+")
    p = commands.add_parser("predict")
    p.add_argument("--reference", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--local-only", action="store_true")
    for p in commands.choices.values():
        p.add_argument("--probe-chunk-size", type=int)
        p.add_argument("--probe-edge-chunk-size", type=int)
        p.add_argument("--spectral-chunk-size", type=int)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"train": train, "finetune": finetune, "evaluate": evaluate, "predict": predict}[
        command
    ](**args)


if __name__ == "__main__":
    main()
