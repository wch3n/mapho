"""Train a joint MACE/FNO PAW pilot without target electronic inputs."""

from ..schema import canonical_schema

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from ..field_model import FieldModelConfig, mace_graph, probe_basis, resize_field
from ..probes import PeriodicProbeGrid
from .model import PAWModel, atomic_inputs, point_basis, periodic_interpolate
from .paw import sha, save_json


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def paw_contract(operator):
    channels = json.loads(operator["paw"].attrs["channels_json"])
    numbers = operator["atomic_numbers"][:]
    types = operator["paw/types"][:]
    mapping = {}
    mask = np.zeros(operator["paw/d"].shape, dtype=bool)
    for i, (z, t) in enumerate(zip(numbers, types)):
        ls = channels[t]
        if any(l not in (0, 1, 2) for l in ls):
            raise ValueError("Only complete s/p/d projectors supported")
        if int(z) in mapping and mapping[int(z)] != ls:
            raise ValueError("Multiple PAW datasets for one element")
        mapping[int(z)] = ls
        n = sum(2 * l + 1 for l in ls)
        mask[i, :n, :n] = True
    if max(sum(2 * l + 1 for l in ls) for ls in mapping.values()) != mask.shape[-1]:
        raise ValueError("Unexpected native padding dimension")
    return mapping, mask


def onsite_mse(prediction, item):
    squared = (prediction - item["target_d"]).square()
    return squared[item["d_mask"]].mean() if "d_mask" in item else squared.mean()


def target_contract(target, baseline, target_path, dataset, manifest, target_kind):
    """Require an explicit conditional target and the matching reference exchange."""
    if canonical_schema(baseline.attrs["schema"]) != "mapho.vasp_paw_operator.v1":
        raise ValueError("Expected a scalar PBE atomic baseline")
    schema = canonical_schema(target.attrs["schema"])
    if target_kind == "pbe":
        if schema != "mapho.vasp_paw_operator.v1":
            raise ValueError("Expected scalar PBE PAW operator schemas")
        return {}
    if target_kind != "pbe0_components":
        raise ValueError("Target kind must be pbe or pbe0_components")
    from .ace import SCHEMA as ACE_SCHEMA, _require_pbe0
    from .paw import COMPONENT_SCHEMA

    if schema != COMPONENT_SCHEMA:
        raise ValueError(
            "Conditional PBE0 training requires explicitly marked components"
        )
    relative = target_path.relative_to(dataset).as_posix()
    paired = manifest.get("hybrid_operators", {}).get(relative)
    if paired is None:
        raise ValueError("A paired reference exchange archive is required")
    operator = dataset / paired
    with h5py.File(operator) as archive:
        if archive.attrs["schema"] != ACE_SCHEMA:
            raise ValueError("A complete reference PAW+ACE archive is required")
        meta = json.loads(archive.attrs["metadata_json"])
    _require_pbe0(meta["reference_manifest"]["settings"])
    if Path(meta["components"]).resolve() != target_path.resolve() or meta[
        "components_sha256"
    ] != sha(target_path):
        raise ValueError("Reference exchange belongs to different component targets")
    if sha(meta["ace_file"]) != meta["ace_sha256"]:
        raise ValueError("Reference exchange changed")
    return dict(
        reference_exchange_operator=str(operator),
        reference_exchange_operator_sha256=sha(operator),
    )


def prepare(
    dataset,
    atomic,
    output,
    grid_size=32,
    point_count=8192,
    grid_shape=None,
    *,
    target_kind="pbe",
    correction_activation="silu",
):

    if target_kind not in ("pbe", "pbe0_components") or correction_activation not in (
        "silu",
        "gelu",
    ):
        raise ValueError("Unsupported target kind or activation")
    dataset, atomic, output = map(
        lambda x: Path(x).resolve(), (dataset, atomic, output)
    )
    m = json.loads((dataset / "manifest.json").read_text())
    if (
        not json.loads((dataset / "data_audit.json").read_text())["passed"]
        or not json.loads((atomic / "completion.json").read_text())["complete"]
    ):
        raise ValueError(
            "Audited DFT data and completed atomic references are required"
        )
    am = json.loads((atomic / "manifest.json").read_text())
    if am["dataset_manifest_sha256"] != sha(dataset / "manifest.json"):
        raise ValueError("Atomic references belong to a different dataset")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cache").mkdir()
    (output / "logs").mkdir()
    with h5py.File(dataset / m["splits"]["train"][0]) as f:
        species_channels, _ = paw_contract(f)
    legacy = species_channels == {14: [0, 0, 1, 1]}
    config = FieldModelConfig(
        atomic_numbers=tuple(sorted(species_channels)),
        grid_shape=grid_shape,
        channels=8,
        radial_count=12,
        lmax=2,
        correlation=3,
        grid_size=grid_size,
        fno_width=8,
        fno_layers=2,
        probe_chunk_size=8192,
        probe_edge_chunk_size=32768,
    )
    statistics = []
    entries = {}
    all_hashes = set()
    for split, paths in m["splits"].items():
        entries[split] = []
        for relative in paths:
            target = dataset / relative
            name = target.parent.name
            baseline = atomic / name / "operator.h5"
            with h5py.File(target) as t, h5py.File(baseline) as b:
                meta = json.loads(b.attrs["metadata_json"])
                assert meta["reference_manifest"]["settings"]["ICHARG"] == 12
                contract_metadata = target_contract(
                    t, b, target, dataset, m, target_kind
                )
                for key in (
                    "cell_A",
                    "positions_A",
                    "atomic_numbers",
                    "paw/q",
                    "paw/types",
                ):
                    if not np.allclose(t[key][:], b[key][:], atol=1e-10, rtol=0):
                        raise ValueError("Baseline and target geometry/PAW mismatch")
                contract, d_mask = paw_contract(t)
                if (
                    contract != species_channels
                    or paw_contract(b)[0] != species_channels
                ):
                    raise ValueError("PAW channel contract changed")
                numbers = t["atomic_numbers"][:]
                positions = t["positions_A"][:]
                cell = t["cell_A"][:]
                geometry_hash = sha(target.parent / "POSCAR")
                if geometry_hash in all_hashes:
                    raise ValueError("Duplicate geometry across entries")
                all_hashes.add(geometry_hash)
                reference = b["potential_eV"][:]
                delta = t["potential_eV"][:] - reference
                dd = t["paw/d"][:] - b["paw/d"][:]
                rng = np.random.default_rng(70000 + int(name.split("_")[-1]))
                ids = rng.choice(
                    delta.size, size=min(point_count, delta.size), replace=False
                )
                fractions = np.stack(
                    np.unravel_index(ids, delta.shape), axis=-1
                ) / np.array(delta.shape)
                coarse = resize_field(
                    reference,
                    tuple(grid_shape) if grid_shape is not None else (grid_size,) * 3,
                )
                inputs = atomic_inputs(numbers, positions, cell, coarse)
                np.savez_compressed(
                    output / "cache" / f"{name}.npz",
                    numbers=numbers,
                    positions=positions,
                    cell=cell,
                    baseline=coarse.astype("float32"),
                    envelope=inputs.reference_density.astype("float32"),
                    fractions=fractions.astype("float32"),
                    target=delta.ravel()[ids].astype("float32"),
                    target_d=dd.astype("float32"),
                    d_mask=d_mask,
                )
                entry = dict(
                    name=name,
                    target=str(target),
                    baseline=str(baseline),
                    target_sha256=sha(target),
                    baseline_sha256=sha(baseline),
                    cache_sha256=sha(output / "cache" / f"{name}.npz"),
                    native_grid=list(delta.shape),
                    **contract_metadata,
                )
                entries[split].append(entry)
                if split == "train":
                    graph = mace_graph(inputs, config)
                    statistics.append(
                        dict(
                            v2=float(np.mean(delta**2)),
                            d2=float(np.mean(dd[d_mask] ** 2)),
                            neighbors=graph["edge_index"].shape[1] / len(numbers),
                            baseline_mean=float(coarse.mean()),
                            baseline_second=float(np.mean(coarse**2)),
                            envelope_mean=float(inputs.reference_density.mean()),
                            envelope_second=float(np.mean(inputs.reference_density**2)),
                        )
                    )
            print("Prepared", split, name, flush=True)
    mean = np.mean(
        [[s["baseline_mean"], s["envelope_mean"]] for s in statistics], axis=0
    )
    second = np.mean(
        [[s["baseline_second"], s["envelope_second"]] for s in statistics], axis=0
    )
    field_scale = float(np.mean([s["v2"] for s in statistics]) ** 0.5)
    metadata = dict(
        schema="mapho.vasp_training.v1",
        dataset=str(dataset),
        atomic_reference=str(atomic),
        target_kind=target_kind,
        correction_activation=correction_activation,
        conditional_reference_exchange=target_kind == "pbe0_components",
        prediction_scope="Smooth potential and PAW D; reference ACE exchange required"
        if target_kind == "pbe0_components"
        else "PBE smooth potential and PAW D",
        config=asdict(config),
        entries=entries,
        point_count=point_count,
        species_channels=None if legacy else species_channels,
        onsite_loss="active native matrix entries only",
        field_scale=field_scale,
        onsite_scale=float(np.mean([s["d2"] for s in statistics]) ** 0.5),
        avg_neighbors=float(np.mean([s["neighbors"] for s in statistics])),
        input_mean=[0.0, *mean.tolist()],
        input_scale=[field_scale, *np.maximum(second - mean**2, 1e-12) ** 0.5],
        normalization_split="train only",
        dataset_manifest_sha256=sha(dataset / "manifest.json"),
        baseline_definition="VASP ICHARG=12 fixed superposed atomic densities, no SCF input files",
        native_point_loss=True,
        fno_interpolation="periodic trilinear, same at training and inference",
        seed=31,
        modes=["local", "full"]
        if target_kind == "pbe0_components"
        else ["local", "none", "full"],
    )
    save_json(output / "manifest.json", metadata)
    return metadata


def load_items(root, split, config, device):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    result = []
    for entry in m["entries"][split]:
        path = root / "cache" / f"{entry['name']}.npz"
        if sha(path) != entry["cache_sha256"]:
            raise ValueError("Training cache changed")
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
        inputs = atomic_inputs(
            data["numbers"], data["positions"], data["cell"], data["baseline"]
        )
        graph = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in mace_graph(inputs, config).items()
        }
        item = dict(
            name=entry["name"],
            graph=graph,
            shape=data["baseline"].shape,
            probes=point_basis(
                inputs,
                np.stack(
                    np.unravel_index(
                        np.arange(data["baseline"].size), data["baseline"].shape
                    ),
                    -1,
                )
                / np.array(data["baseline"].shape),
                config,
            ).to(device),
            point_probes=point_basis(inputs, data["fractions"], config).to(device),
        )
        for key in ("cell", "baseline", "envelope", "fractions", "target", "target_d"):
            item[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
        if "d_mask" in data:
            item["d_mask"] = torch.tensor(
                data["d_mask"], dtype=torch.bool, device=device
            )
        result.append(item)
    return result


def score(model, items):
    model.eval()
    values = []
    with torch.no_grad():
        for item in items:
            field, d = model(item)
            values.append(
                [
                    float((field - item["target"]).square().mean()),
                    float(onsite_mse(d, item)),
                ]
            )
    mses = np.mean(values, axis=0)
    return dict(
        field_rmse_eV=float(mses[0] ** 0.5),
        onsite_rmse_eV=float(mses[1] ** 0.5),
        objective=float(
            mses[0] / float(model.field_scale) ** 2
            + mses[1] / float(model.onsite_scale) ** 2
        ),
    )


def train(root, mode, epochs=300, device="cuda"):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    out = root / mode
    out.mkdir(exist_ok=False)
    config = FieldModelConfig(**m["config"])
    model = PAWModel(
        config,
        mode=mode,
        avg_neighbors=m["avg_neighbors"],
        correction_activation=m.get("correction_activation", "silu"),
        species_channels=m.get("species_channels"),
    ).to(device)
    model.field_scale.fill_(m["field_scale"])
    model.onsite_scale.fill_(m["onsite_scale"])
    if model.correction is not None:
        model.correction.input_mean.copy_(torch.tensor(m["input_mean"], device=device))
        model.correction.input_scale.copy_(
            torch.tensor(m["input_scale"], device=device)
        )
    initial = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    train_items = load_items(root, "train", config, device)
    validation = load_items(root, "validation", config, device)
    groups = [
        dict(
            params=[
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("local.backbone")
            ],
            lr=3e-4,
        ),
        dict(
            params=[
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("local.backbone")
            ],
            lr=1e-3,
        ),
    ]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=8, min_lr=1e-5
    )
    history = [dict(epoch=0, validation=score(model, validation))]
    best = history[0]["validation"]["objective"]
    best_epoch = 0
    best_state = cpu_state(model)
    started = time.monotonic()
    rng = np.random.default_rng(31)
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for index in rng.permutation(len(train_items)):
            item = train_items[index]
            optimizer.zero_grad(set_to_none=True)
            field, d = model(item)
            loss = (
                (field - item["target"]) / model.field_scale
            ).square().mean() + onsite_mse(d, item) / model.onsite_scale.square()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == epochs:
            val = score(model, validation)
            scheduler.step(val["objective"])
            if val["objective"] < best:
                best = val["objective"]
                best_epoch = epoch
                best_state = cpu_state(model)
            history.append(
                dict(
                    epoch=epoch,
                    training_objective=float(np.mean(losses)),
                    validation=val,
                    learning_rates=[g["lr"] for g in optimizer.param_groups],
                )
            )
            save_json(out / "history.json", history)
            torch.save(
                dict(
                    schema="mapho.vasp_paw_model.v1",
                    state_dict=best_state,
                    description=model.description(),
                    training_manifest_sha256=sha(root / "manifest.json"),
                    best_epoch=best_epoch,
                ),
                out / "model.pt",
            )
            print(
                mode,
                "epoch",
                epoch,
                "val",
                val,
                "best",
                best_epoch,
                "seconds",
                round(time.monotonic() - started, 1),
                flush=True,
            )
        if epoch >= 150 and epoch - best_epoch >= 100:
            break
    model.load_state_dict(best_state)
    changed = {}
    for prefix in (
        "local.backbone",
        "local.coefficients",
        "local.offset",
        "onsite_",
        "correction",
    ):
        deltas = [
            (p.detach().cpu() - initial[n]).double()
            for n, p in model.named_parameters()
            if n in initial and n.startswith(prefix)
        ]
        changed[prefix] = dict(
            changed_tensors=sum(bool(torch.count_nonzero(x)) for x in deltas),
            l2=float(sum(x.square().sum() for x in deltas)) ** 0.5,
        )
    report = dict(
        complete=True,
        mode=mode,
        seed=31,
        epochs=epoch,
        updates=epoch * len(train_items),
        best_epoch=best_epoch,
        correction_activation=model.correction_activation,
        validation=score(model, validation),
        training=score(model, train_items),
        parameter_changes=changed,
        wall_seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        cuda_peak_MiB=torch.cuda.max_memory_allocated() / 1024**2
        if device.startswith("cuda")
        else None,
        versions={n: version(n) for n in ("torch", "e3nn", "mace-torch", "numpy")},
    )
    save_json(out / "training.json", report)
    return report


def load_model(root, mode, device):
    root = Path(root)
    checkpoint = torch.load(
        root / mode / "model.pt", map_location="cpu", weights_only=False
    )
    if checkpoint["training_manifest_sha256"] != sha(root / "manifest.json"):
        raise ValueError("Model/data contract changed")
    desc = checkpoint["description"]
    activation = desc.get("correction_activation", "silu")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        activation != manifest.get("correction_activation", "silu")
        or desc["mode"] != mode
    ):
        raise ValueError("Model activation/mode differs from training contract")
    model = PAWModel(
        FieldModelConfig(**desc["config"]),
        mode=mode,
        channels=desc["channels"],
        avg_neighbors=desc["avg_neighbors"],
        correction_activation=activation,
        species_channels=desc.get("species_channels"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def predict(model, entry, device):
    """Read only the geometry/atomic baseline; target path is not accessed."""
    # Match training in fresh processes, including the pointwise Conv3d layers.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha(entry["baseline"]) != entry["baseline_sha256"]:
        raise ValueError("Atomic input changed")
    with h5py.File(entry["baseline"]) as f:
        base = f["potential_eV"][:]
        d0 = f["paw/d"][:]
        numbers = f["atomic_numbers"][:]
        positions = f["positions_A"][:]
        cell = f["cell_A"][:]
    c = model.config
    coarse = resize_field(
        base, tuple(c.grid_shape) if c.grid_shape is not None else (c.grid_size,) * 3
    )
    inputs = atomic_inputs(numbers, positions, cell, coarse)
    graph = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in mace_graph(inputs, c).items()
    }
    with torch.no_grad():
        coeff, offset, dd = model.coefficients(graph)
        native = PeriodicProbeGrid(inputs, base.shape, c).contract(coeff) + offset
        if model.correction is not None:
            low = (
                PeriodicProbeGrid(inputs, coarse.shape, c).contract(coeff) + offset
            ).reshape(coarse.shape)
            channels = torch.stack(
                [
                    low,
                    torch.tensor(coarse, dtype=low.dtype, device=device),
                    torch.tensor(
                        inputs.reference_density, dtype=low.dtype, device=device
                    ),
                ]
            )
            correction = (
                model.correction(
                    channels, torch.tensor(cell, dtype=low.dtype, device=device)
                )
                * model.field_scale
            )
            for start in range(0, base.size, 65536):
                ids = np.arange(start, min(start + 65536, base.size))
                frac = np.stack(np.unravel_index(ids, base.shape), -1) / np.array(
                    base.shape
                )
                native[start : start + len(ids)] += periodic_interpolate(
                    correction, torch.tensor(frac, dtype=low.dtype, device=device)
                )
        return base + native.cpu().numpy().reshape(base.shape), d0 + dd.cpu().numpy()


def evaluate(root, mode, device="cuda"):
    torch.set_num_threads(1)
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    model = load_model(root, mode, device)
    out = root / mode / "predictions"
    out.mkdir(exist_ok=False)
    rows = []
    for entry in m["entries"]["test"]:
        v, d = predict(model, entry, device)
        # Target access begins only after prediction, exclusively for scoring.
        with h5py.File(entry["target"]) as t, h5py.File(entry["baseline"]) as b:
            target = t["potential_eV"][:]
            td = t["paw/d"][:]
            _, mask = paw_contract(t)
            row = dict(
                name=entry["name"],
                field_rmse_eV=float(np.mean((v - target) ** 2) ** 0.5),
                onsite_rmse_eV=float(np.mean((d[mask] - td[mask]) ** 2) ** 0.5),
                baseline_field_rmse_eV=float(
                    np.mean((b["potential_eV"][:] - target) ** 2) ** 0.5
                ),
                baseline_onsite_rmse_eV=float(
                    np.mean((b["paw/d"][:][mask] - td[mask]) ** 2) ** 0.5
                ),
            )
        np.savez_compressed(out / f"{entry['name']}.npz", potential=v, d=d)
        rows.append(row)
        print(mode, "test", row, flush=True)
    report = dict(
        mode=mode,
        rows=rows,
        **{
            k: float(np.mean([r[k] ** 2 for r in rows]) ** 0.5)
            for k in rows[0]
            if k != "name"
        },
    )
    save_json(root / mode / "test_fields.json", report)
    return report
