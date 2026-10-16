"""Structure-level splits, deterministic training, and safe tensor checkpoints."""

import json
import importlib.metadata
from pathlib import Path
import platform
import random

import numpy as np
import torch

from .graph import batch_graphs
from .model import HamiltonianModel, ModelConfig, orbital_irreps


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_samples(samples, config):
    for sample in samples:
        sample.validate()
        if orbital_irreps(sample.basis) != orbital_irreps(config.basis):
            raise ValueError("Sample and model orbital bases differ")
        if abs(sample.graph.cutoff - config.cutoff) > 1e-7:
            raise ValueError("Sample and model cutoffs differ")
        if sample.hamiltonian.is_complex():
            raise ValueError(
                "This learner requires explicitly real spinless labels; complex data are not discarded"
            )
    references = {s.metadata.get("energy_reference") for s in samples}
    if len(references) != 1 or not next(iter(references)):
        raise ValueError(
            "All samples must declare the same nonempty energy_reference convention"
        )
    orthogonality = {s.metadata.get("orthogonal_basis") for s in samples}
    if len(orthogonality) != 1 or not all(
        type(s.metadata.get("orthogonal_basis")) is bool for s in samples
    ):
        raise ValueError("All samples must declare the same boolean orthogonal_basis")


def _pack(samples, device, dtype=torch.float32):
    graph = batch_graphs([s.graph for s in samples]).to(device, dtype=dtype)
    targets = torch.cat([s.hamiltonian for s in samples]).to(device=device, dtype=dtype)
    return graph, targets


@torch.no_grad()
def evaluate(model, samples, device="cpu"):
    model.eval()
    absolute = squared = count = 0
    on_abs = off_abs = on_count = off_count = 0
    for start in range(0, len(samples), 16):
        graph, targets = _pack(
            samples[start : start + 16], device, next(model.parameters()).dtype
        )
        error = model(graph) - targets
        absolute += error.abs().sum().item()
        squared += error.square().sum().item()
        count += error.numel()
        on_abs += error[graph.onsite].abs().sum().item()
        off_abs += error[~graph.onsite].abs().sum().item()
        on_count += error[graph.onsite].numel()
        off_count += error[~graph.onsite].numel()
    return {
        "mae_eV": absolute / count,
        "rmse_eV": (squared / count) ** 0.5,
        "onsite_mae_eV": on_abs / max(on_count, 1),
        "offsite_mae_eV": off_abs / max(off_count, 1),
    }


def fit(
    samples,
    output,
    config=None,
    epochs=80,
    batch_size=8,
    learning_rate=0.003,
    seed=7,
    device="cpu",
    sample_ids=None,
):
    """Reserve validation and test structures; select checkpoints only on validation."""
    config = config or ModelConfig()
    if (
        len(samples) < 8
        or epochs < 1
        or batch_size < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "Need >=8 structures, positive epochs, batch size, and learning rate"
        )
    _validate_samples(samples, config)
    sample_ids = (
        list(sample_ids)
        if sample_ids is not None
        else [str(i) for i in range(len(samples))]
    )
    if len(sample_ids) != len(samples) or len(set(sample_ids)) != len(samples):
        raise ValueError("Sample identifiers must be unique and match the dataset")
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    order = np.random.default_rng(seed).permutation(len(samples))
    n_holdout = max(1, int(0.2 * len(samples)))
    split = {
        "test": order[:n_holdout],
        "validation": order[n_holdout : 2 * n_holdout],
        "train": order[2 * n_holdout :],
    }
    partitions = {
        name: [samples[i] for i in indices] for name, indices in split.items()
    }
    (output / "split.json").write_text(
        json.dumps(
            {name: [sample_ids[i] for i in indices] for name, indices in split.items()},
            indent=2,
        )
        + "\n"
    )
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "device": str(device),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "e3nn", "ase", "numpy", "h5py")
                },
            },
            indent=2,
        )
        + "\n"
    )
    # Species in holdouts must already occur in training.
    species = sorted({int(z) for s in partitions["train"] for z in s.graph.numbers})
    if any(int(z) not in species for s in samples for z in s.graph.numbers):
        raise ValueError(
            "Holdout contains unseen elements; choose a stratified dataset/split"
        )
    model = HamiltonianModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial = evaluate(model, partitions["validation"], device)
    best, history = float("inf"), []
    generator = np.random.default_rng(seed + 1)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = generator.permutation(len(partitions["train"]))
        loss_sum, steps = 0.0, 0
        for start in range(0, len(permutation), batch_size):
            batch = [
                partitions["train"][i] for i in permutation[start : start + batch_size]
            ]
            graph, targets = _pack(batch, device)
            prediction = model(graph)
            # Balance onsite/offsite categories instead of overwhelming onsite
            # supervision with the many more hopping blocks.
            error = (prediction - targets).square()
            losses = [
                error[mask].mean()
                for mask in (graph.onsite, ~graph.onsite)
                if mask.any()
            ]
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
        validation = evaluate(model, partitions["validation"], device)
        history.append(
            {"epoch": epoch, "train_balanced_mse": loss_sum / steps, **validation}
        )
        if validation["mae_eV"] < best:
            best = validation["mae_eV"]
            torch.save(
                {
                    "schema_version": 1,
                    "config": config.as_dict(),
                    "state_dict": model.state_dict(),
                    "species": species,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_mae_eV": best,
                    "energy_reference": samples[0].metadata["energy_reference"],
                    "orthogonal_basis": samples[0].metadata["orthogonal_basis"],
                },
                output / "model.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}  validation MAE {best:.6f} eV (best)", flush=True)
    model, checkpoint = load_checkpoint(output / "model.pt", device)
    report = {
        "initial_validation": initial,
        "best_epoch": checkpoint["epoch"],
        "validation": evaluate(model, partitions["validation"], device),
        "test": evaluate(model, partitions["test"], device),
        "config": config.as_dict(),
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "counts": {name: len(values) for name, values in partitions.items()},
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, report, partitions


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model = HamiltonianModel(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
