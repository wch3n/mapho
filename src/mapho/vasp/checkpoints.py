"""Load conditioned PAW checkpoints with their original provenance contracts."""

import json
from pathlib import Path

import torch

from .paw import sha
from .paw_conditioning import ConditionedPAWModel
from .train import load_model as load_base


def read_json(path):
    return json.loads(Path(path).read_text())


def check_hash(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Changed provenance input: {path}")


def load_base_for_repair(manifest, label, device):
    spec = manifest["models"][label]
    root = Path(spec["root"])
    check_hash(root / "manifest.json", spec["manifest_sha256"])
    check_hash(root / spec["mode"] / "model.pt", spec["checkpoint_sha256"])
    return load_base(root, spec["mode"], device)


def load_repaired_model(root, label, device="cuda"):
    root = Path(root)
    m = read_json(root / "manifest.json")
    if "trained_root" in m:
        trained = Path(m["trained_root"])
        check_hash(trained / "manifest.json", m["trained_manifest_sha256"])
        check_hash(trained / label / "model.pt", m["trained_checkpoints"][label])
        return load_repaired_model(trained, label, device)
    path = root / label / "model.pt"
    check_hash(path, read_json(root / label / "training.json")["checkpoint_sha256"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint["manifest_sha256"] != sha(root / "manifest.json")
        or checkpoint["label"] != label
    ):
        raise ValueError("Repair contract changed")
    model = (
        ConditionedPAWModel(
            load_base_for_repair(m, label, device), checkpoint["specifications"]
        )
        .to(device)
        .eval()
    )
    model.heads.load_state_dict(checkpoint["heads"])
    return model


def initial_joint_model(manifest, label, device):
    parent = Path(manifest["repaired"])
    check_hash(parent / "manifest.json", manifest["repaired_manifest_sha256"])
    check_hash(parent / label / "model.pt", manifest["parent_checkpoints"][label])
    frozen = load_repaired_model(parent, label, device)
    model = ConditionedPAWModel(frozen.base, frozen.specifications, train_base=True).to(
        device
    )
    model.heads.load_state_dict(frozen.heads.state_dict())
    return model


def load_joint_model(root, label, device="cuda"):
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
    model = initial_joint_model(m, label, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
