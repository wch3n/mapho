"""Run validated PBE or conditional PBE0 PAW training from a YAML file."""

from hashlib import sha256
from pathlib import Path

import yaml

from ...vasp.training_config import (
    TRAINING_DEFAULTS,
    integer,
    mapping,
    model_settings,
    sampling_settings,
    training_settings,
)


class UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys, including keys introduced through YAML merges."""


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def path_value(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path, overrides=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        data = yaml.load(raw, Loader=UniqueSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML: {exc}") from exc
    data = mapping(
        data,
        "configuration",
        {
            "version",
            "dataset",
            "atomic_reference",
            "output",
            "target",
            "mode",
            "correction_activation",
            "model",
            "sampling",
            "training",
        },
    )
    if integer(data.get("version", 1), "version") != 1:
        raise ValueError("Only training configuration version 1 is supported")
    for section in ("model", "sampling", "training"):
        if section in data and not isinstance(data[section], dict):
            raise ValueError(f"{section} must be a mapping")
    overrides = mapping(
        {} if overrides is None else overrides,
        "CLI overrides",
        {
            "output",
            "mode",
            "epochs",
            "device",
            "seed",
            "backbone_lr",
            "head_lr",
        },
    )
    for key in ("dataset", "output"):
        if key not in data and key not in overrides:
            raise ValueError(f"Missing required configuration key: {key}")
    dataset = path_value(data["dataset"], path.parent, "dataset")
    atomic = path_value(
        data.get("atomic_reference", str(Path(dataset) / "atomic")),
        path.parent,
        "atomic_reference",
    )
    # YAML paths are relative to the YAML file. CLI paths are relative to cwd.
    output = path_value(
        overrides.get("output", data.get("output")),
        Path.cwd() if "output" in overrides else path.parent,
        "output",
    )
    target = data.get("target", "pbe")
    mode = overrides.get("mode", data.get("mode", "full"))
    activation = data.get("correction_activation", "gelu")
    if target not in ("pbe", "pbe0_components"):
        raise ValueError("target must be pbe or pbe0_components")
    if mode not in ("local", "none", "full") or (
        target == "pbe0_components" and mode == "none"
    ):
        raise ValueError("mode must be local/full for PBE0, or local/none/full for PBE")
    if activation not in ("silu", "gelu"):
        raise ValueError("correction_activation must be silu or gelu")
    if target == "pbe0_components" and activation != "gelu":
        raise ValueError("The conditional PBE0 pilot requires GELU")
    training = mapping(data.get("training", {}), "training", TRAINING_DEFAULTS.keys())
    training.update({k: v for k, v in overrides.items() if k in TRAINING_DEFAULTS})
    return dict(
        version=1,
        dataset=dataset,
        atomic_reference=atomic,
        output=output,
        target=target,
        mode=mode,
        correction_activation=activation,
        model=model_settings(data.get("model", {})),
        sampling=sampling_settings(data.get("sampling", {})),
        training=training_settings(training),
        source_config=str(path),
        source_config_sha256=sha256(raw).hexdigest(),
    )


def run(path, *, overrides=None, prepare_only=False, dry_run=False):
    settings = load_config(path, overrides)
    if dry_run:
        return settings
    from ...vasp import train as training
    from ...vasp.paw import save_json, sha

    source = Path(settings["source_config"]).read_bytes()
    if sha256(source).hexdigest() != settings["source_config_sha256"]:
        raise ValueError("Configuration changed while preparing the run")
    dataset = Path(settings["dataset"])
    if settings["target"] == "pbe0_components":
        from .pbe0_learning import audit

        if not audit(dataset)["passed"]:
            raise ValueError(
                "Every PBE0 reference and atomic baseline must pass before training"
            )
    root = Path(settings["output"])
    manifest = training.prepare(
        dataset,
        settings["atomic_reference"],
        root,
        point_count=settings["sampling"]["point_count"],
        sampling_seed=settings["sampling"]["seed"],
        target_kind=settings["target"],
        correction_activation=settings["correction_activation"],
        model_options=settings["model"],
        training_options=settings["training"],
    )
    (root / "config.input.yaml").write_bytes(source)
    save_json(root / "config.resolved.json", settings)
    manifest["configuration_files"] = {
        name: sha(root / name) for name in ("config.input.yaml", "config.resolved.json")
    }
    save_json(root / "manifest.json", manifest)
    if not prepare_only:
        training.train(root, settings["mode"])
        training.evaluate(root, settings["mode"], device=settings["training"]["device"])
    return dict(
        output=str(root),
        mode=settings["mode"],
        prepared=True,
        trained=not prepare_only,
        conditional_reference_exchange=settings["target"] == "pbe0_components",
    )
