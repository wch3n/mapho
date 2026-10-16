"""Reproducible small insulating silicon dataset, generated in the QE runtime."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write

from .backend import QEConfig, run_reference
from .upf import read_upf_metadata


def silicon_dataset(output, pseudo, count=32, seed=29):
    if count < 10:
        raise ValueError("Use at least ten independent structures")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    config = QEConfig()
    order = rng.permutation(count)
    ntrain, nval = int(count * 0.625), int(count * 0.1875)
    splits = {
        "train": order[:ntrain],
        "validation": order[ntrain : ntrain + nval],
        "test": order[ntrain + nval :],
    }
    manifest = {
        "schema": "mapho.field_dataset.v1",
        "seed": seed,
        "distribution": "diamond Si: isotropic strain +/-1.5%, symmetric strain perturbation +/-0.6%, second-atom displacement +/-0.025 angstrom per axis",
        "qe_config": asdict(config),
        "pseudo": read_upf_metadata(pseudo),
        "splits": {
            k: [f"sample_{i:03d}/fields.npz" for i in v] for k, v in splits.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    structures = []
    for i in range(count):
        atoms = bulk("Si", "diamond", a=5.43)
        strain = rng.uniform(-0.006, 0.006, (3, 3))
        strain = (strain + strain.T) / 2
        strain += np.eye(3) * rng.uniform(-0.015, 0.015)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        atoms.positions[1] += rng.uniform(-0.025, 0.025, 3)
        atoms.info["sample_id"] = i
        structures.append(atoms)
        sample = run_reference(atoms, {"Si": pseudo}, out / f"sample_{i:03d}", config)
        occupied = sample.occupations > 1
        gap = float(
            sample.eigenvalues[~occupied].min() - sample.eigenvalues[occupied].max()
        )
        if gap <= 0:
            raise RuntimeError(
                f"Noninsulating sample {i}: gap={gap}; fixed occupations invalid"
            )
        print(
            f"{i + 1}/{count}: grid={sample.shape}, sampled gap={gap:.4f} eV",
            flush=True,
        )
    write(out / "structures.extxyz", structures)
    (out / "complete.json").write_text(json.dumps({"count": count}) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pseudo", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    silicon_dataset(args.output, args.pseudo, args.count, args.seed)


if __name__ == "__main__":
    main()
