"""Validated settings shared by the PAW training API and configuration runner."""

from dataclasses import asdict
import math
import re


TRAINING_DEFAULTS = dict(
    epochs=300,
    device="cuda",
    seed=31,
    backbone_lr=3e-4,
    head_lr=1e-3,
    validation_interval=5,
    scheduler_factor=0.5,
    scheduler_patience=8,
    min_lr=1e-5,
    gradient_clip=5.0,
    early_stopping_min_epochs=150,
    early_stopping_patience=100,
)
SAMPLING_DEFAULTS = dict(point_count=8192, seed=70000)


def mapping(value, name, allowed):
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    return dict(value)


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def positive(value, name):
    # YAML 1.1 reads some scientific notation (e.g. 1e-3) as strings.
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def training_settings(options=None):
    result = dict(TRAINING_DEFAULTS)
    result.update(
        mapping({} if options is None else options, "training", result.keys())
    )
    for key in ("epochs", "validation_interval", "early_stopping_patience"):
        result[key] = integer(result[key], "training." + key)
    for key in ("seed", "scheduler_patience", "early_stopping_min_epochs"):
        result[key] = integer(result[key], "training." + key, 0)
    for key in (
        "backbone_lr",
        "head_lr",
        "scheduler_factor",
        "min_lr",
        "gradient_clip",
    ):
        result[key] = positive(result[key], "training." + key)
    if result["scheduler_factor"] >= 1:
        raise ValueError("training.scheduler_factor must be less than one")
    if result["min_lr"] > min(result["backbone_lr"], result["head_lr"]):
        raise ValueError("training.min_lr must not exceed either initial learning rate")
    if not isinstance(result["device"], str) or not re.fullmatch(
        r"cpu|cuda(?::[0-9]+)?", result["device"]
    ):
        raise ValueError("training.device must be cpu, cuda, or cuda:N")
    return result


def sampling_settings(options=None):
    result = dict(SAMPLING_DEFAULTS)
    result.update(
        mapping({} if options is None else options, "sampling", result.keys())
    )
    result["point_count"] = integer(result["point_count"], "sampling.point_count")
    result["seed"] = integer(result["seed"], "sampling.seed", 0)
    return result


def model_settings(options=None, *, grid_size=32, grid_shape=None):
    from ..field_model import FieldModelConfig

    defaults = asdict(
        FieldModelConfig(
            grid_size=grid_size,
            grid_shape=grid_shape,
            probe_chunk_size=8192,
            probe_edge_chunk_size=32768,
        )
    )
    defaults.pop("atomic_numbers")  # Inferred and checked against the PAW datasets.
    result = dict(defaults)
    result.update(mapping({} if options is None else options, "model", defaults.keys()))
    for key in (
        "channels",
        "correlation",
        "num_interactions",
        "fno_width",
        "fno_layers",
        "probe_chunk_size",
        "probe_edge_chunk_size",
        "spectral_chunk_size",
    ):
        result[key] = integer(result[key], "model." + key)
    result["radial_count"] = integer(result["radial_count"], "model.radial_count", 2)
    result["grid_size"] = integer(result["grid_size"], "model.grid_size", 4)
    result["lmax"] = integer(result["lmax"], "model.lmax", 0)
    for key in ("atom_cutoff", "probe_cutoff", "q_cutoff"):
        result[key] = positive(result[key], "model." + key)
    shape = result["grid_shape"]
    if shape is not None:
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError("model.grid_shape must have three integer dimensions")
        result["grid_shape"] = [integer(n, "model.grid_shape", 4) for n in shape]
    FieldModelConfig(**result).validate()
    return result
