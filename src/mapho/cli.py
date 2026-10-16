"""Command-line demonstration, dataset training, and structure-only prediction."""

import argparse
import json
from pathlib import Path

from ase.io import read
import numpy as np
import torch

from .bands import band_energies
from .data import load_sample, save_sample
from .graph import make_graph
from .model import ModelConfig
from .synthetic import synthetic_sample
from .train import fit, load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Equivariant localized-orbital Hamiltonian learning"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--epochs", type=int, default=80)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--channels", type=int, default=8)
        command.add_argument("--layers", type=int, default=2)
        command.add_argument("--learning-rate", type=float, default=0.003)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
        if name == "demo":
            command.add_argument("--samples", type=int, default=40)
            command.add_argument("--basis", choices=("s", "sp", "spd"), default="sp")
            command.add_argument("--cutoff", type=float, default=4.0)
        else:
            command.add_argument("--data", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--structure", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--kpoints", type=Path, help="Text file with fractional reciprocal k points"
    )
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.command == "predict":
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        atoms = read(args.structure)
        unknown = sorted(set(map(int, atoms.numbers)) - set(checkpoint["species"]))
        if unknown:
            raise ValueError(f"Elements absent from training: {unknown}")
        graph = make_graph(atoms, model.config.cutoff).to(args.device)
        k = np.loadtxt(args.kpoints, ndmin=2) if args.kpoints else np.zeros((1, 3))
        orthogonal = checkpoint.get("orthogonal_basis", True)
        if not orthogonal and args.kpoints:
            raise ValueError(
                "Nonorthogonal bands require supplied overlap blocks via the band_energies Python API"
            )
        with torch.no_grad():
            h = model(graph)
            band_output = {}
            if orthogonal:
                band_output = {
                    "kpoints": k,
                    "band_energies_eV": band_energies(h, graph, k).cpu().numpy(),
                }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(
                stream,
                pairs=graph.pairs.cpu().numpy(),
                hamiltonian=h.cpu().numpy(),
                orthogonal_basis=orthogonal,
                basis=str(model.orbitals),
                cell=atoms.cell.array,
                positions=atoms.positions,
                numbers=atoms.numbers,
                **band_output,
            )
        print(args.output)
        return
    if args.command == "demo":
        if args.samples < 8:
            raise ValueError("The demo needs at least eight structures")
        samples = [
            synthetic_sample(args.seed + i, args.basis, args.cutoff)
            for i in range(args.samples)
        ]
        sample_ids = [f"synthetic_seed_{args.seed + i}" for i in range(args.samples)]
    else:
        paths = sorted(args.data.glob("*.npz"))
        samples = [load_sample(path) for path in paths]
        sample_ids = [str(path.resolve()) for path in paths]
        if not samples:
            raise ValueError("No NPZ training samples found")
    config = ModelConfig(
        basis=samples[0].basis,
        cutoff=samples[0].graph.cutoff,
        channels=args.channels,
        layers=args.layers,
    )
    model, report, partitions = fit(
        samples,
        args.output,
        config,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
        args.device,
        sample_ids,
    )
    if args.command == "demo":
        sample = partitions["test"][0]
        save_sample(args.output / "heldout_sample.npz", sample)
        k = np.column_stack((np.linspace(0, 0.5, 51), np.zeros((51, 2))))
        graph = sample.graph.to(args.device)
        with torch.no_grad():
            predicted = band_energies(model(graph), graph, k).cpu().numpy()
            reference = (
                band_energies(sample.hamiltonian.to(args.device), graph, k)
                .cpu()
                .numpy()
            )
        np.savez_compressed(
            args.output / "heldout_bands.npz",
            kpoints=k,
            predicted_eV=predicted,
            reference_eV=reference,
        )
        report["one_heldout_structure_band_mae_eV"] = float(
            np.abs(predicted - reference).mean()
        )
        report["dataset"] = "analytic synthetic labels, not DFT"
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
