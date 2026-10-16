"""Predict primitive Si bands with one YAML-trained PBE PAW checkpoint.

Run inside a GPU allocation with the runtime of the private VASP executable.
The atomic-input prediction is completed before either PBE reference is run.
"""

import argparse
import csv
import json
from pathlib import Path
import shlex

from ase.build import bulk
import h5py
import numpy as np

from mapho.experiments.vasp.primitive_bands import silicon_path
from mapho.experiments.vasp.silicon_dataset import case_report, compare_operators
from mapho.experiments.vasp.training_validation import LAUNCHER
from mapho.vasp.paw import (
    compare,
    export_operator,
    prepare_reference,
    prepare_replay,
    read_bands,
    run,
    save_json,
    sha,
    write_incar,
)
from mapho.vasp.train import load_model, predict


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare_case(folder, atoms, potcar, encut, mesh, *, atomic=False, shape=None):
    manifest = prepare_reference(
        folder, atoms, {"Si": potcar}, encut=encut, mesh=mesh, nbands=8
    )
    settings = manifest["settings"]
    settings.update(NELMIN=16, LWAVE=False, LCHARG=False)
    if atomic:
        settings.update(ICHARG=12, NELMDL=0)
    if shape is not None:
        settings.update(dict(zip(("NGXF", "NGYF", "NGZF"), shape)))
    write_incar(folder / "INCAR", settings)
    manifest["inputs"] = {
        name: sha(folder / name) for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
    }
    save_json(folder / "manifest.json", manifest)


def write_results(root, path, reference, predicted, mode):
    """Export every level and plot both spectra with one reference energy zero."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    shape = (len(path["kpoints"]), 8)
    if any(
        e.shape != shape or not np.isfinite(e).all() for e in (reference, predicted)
    ):
        raise ValueError("Expected eight finite bands at every path point")
    delta = predicted - reference
    metrics = dict(
        frontier_mae_meV=float(np.abs(delta[:, 2:6]).mean() * 1000),
        frontier_max_meV=float(np.abs(delta[:, 2:6]).max() * 1000),
        pbe_path_gap_eV=float(reference[:, 4].min() - reference[:, 3].max()),
        predicted_path_gap_eV=float(predicted[:, 4].min() - predicted[:, 3].max()),
        frontier_definition="two occupied and two empty bands, all path points",
        energy_alignment="one PBE VBM zero for plotting only; errors use raw energies",
    )
    save_json(root / "metrics.json", metrics)
    np.savez_compressed(
        root / "bands.npz",
        kpoints=path["kpoints"],
        distance_inv_A=path["distance_inv_A"],
        reference_eV=reference,
        predicted_eV=predicted,
    )
    with (root / "band_source_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["k_index", "kx", "ky", "kz", "band", "pbe_eV", "mapho_eV", "error_meV"]
        )
        for k, point in enumerate(path["kpoints"]):
            for band in range(8):
                writer.writerow(
                    [
                        k,
                        *point,
                        band + 1,
                        reference[k, band],
                        predicted[k, band],
                        1000 * delta[k, band],
                    ]
                )
    x = np.asarray(path["distance_inv_A"])
    zero = reference[:, 3].max()
    color = "#0072B2" if mode == "full" else "#D55E00"
    label = "MAPHO-FNO" if mode == "full" else "MAPHO-MACE"
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.2, 5.6),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    for band in range(8):
        top.plot(
            x,
            reference[:, band] - zero,
            color="0.5",
            ls=(0, (3, 2)),
            lw=0.9,
            label="PBE" if band == 0 else None,
        )
        top.plot(
            x,
            predicted[:, band] - zero,
            color=color,
            lw=0.9,
            label=label if band == 0 else None,
        )
    bottom.plot(x, 1000 * delta[:, 3], color=color, lw=1, label="Top occupied")
    bottom.plot(x, 1000 * delta[:, 4], color=color, lw=1, ls="--", label="Bottom empty")
    for ax in (top, bottom):
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.axvline(x[path["vertex_indices"][1]], color="0.8", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.legend(frameon=False, fontsize=8)
    top.set_ylabel("Energy − PBE VBM (eV)")
    bottom.set_ylabel("Error (meV)")
    bottom.set_xticks(x[path["vertex_indices"]], ["X", "Γ", "L"])
    bottom.set_xlim(x[0], x[-1])
    bottom.set_xlabel("Wave vector")
    fig.savefig(root / "bands.pdf")
    fig.savefig(root / "bands.png", dpi=200)
    plt.close(fig)
    return metrics


def execute(args):
    training, root = args.training.resolve(), args.output.resolve()
    manifest = read_json(training / "manifest.json")
    if manifest.get("target_kind", "pbe") != "pbe" or manifest["config"][
        "atomic_numbers"
    ] != [14]:
        raise ValueError("This example requires a scalar PBE Si checkpoint")
    dataset = read_json(Path(manifest["dataset"]) / "manifest.json")
    executable = Path(dataset["executable"])
    if sha(executable) != dataset["executable_sha256"]:
        raise ValueError("The recorded VASP executable changed")
    entry = manifest["entries"]["train"][0]
    baseline = Path(entry["baseline"])
    if sha(baseline) != entry["baseline_sha256"]:
        raise ValueError("The training atomic reference changed")
    with h5py.File(baseline) as archive:
        metadata = json.loads(archive.attrs["metadata_json"])
    potcar = Path(metadata["reference"]) / "POTCAR"
    native = metadata["reference_manifest"]
    if sha(potcar) != native["inputs"]["POTCAR"]:
        raise ValueError("The PAW dataset changed")
    model = load_model(training, args.mode, args.device)
    root.mkdir(parents=True, exist_ok=False)
    launcher = shlex.split(args.launcher)
    atoms = bulk("Si", "diamond", a=5.43)
    path = silicon_path(atoms.cell.array)
    save_json(
        root / "manifest.json",
        dict(
            training=str(training),
            training_manifest_sha256=sha(training / "manifest.json"),
            checkpoint_sha256=sha(training / args.mode / "model.pt"),
            mode=args.mode,
            executable=str(executable),
            executable_sha256=sha(executable),
            potcar_sha256=sha(potcar),
            example_sha256=sha(__file__),
            lattice_constant_A=5.43,
            reference_meshes=args.meshes,
            path=path,
            band_reference="frozen exported PBE operator; ICHARG=14",
        ),
    )
    atomic = root / "atomic"
    prepare_case(
        atomic, atoms, potcar, native["settings"]["ENCUT"], (1, 1, 1), atomic=True
    )
    run(atomic, executable, launcher=launcher, timeout=3600)
    export_operator(atomic, atomic / "operator.h5")
    with h5py.File(atomic / "operator.h5") as archive:
        shape = archive["potential_eV"].shape
    potential, d = predict(
        model,
        dict(
            baseline=str(atomic / "operator.h5"),
            baseline_sha256=sha(atomic / "operator.h5"),
        ),
        args.device,
    )
    np.savez_compressed(root / "prediction.npz", potential=potential, d=d)
    del model

    # Independent labels are generated only after the prediction is saved.
    references = []
    for mesh in args.meshes:
        folder = root / f"reference_{mesh}"
        prepare_case(
            folder, atoms, potcar, native["settings"]["ENCUT"], (mesh,) * 3, shape=shape
        )
        run(folder, executable, launcher=launcher, timeout=3600)
        export_operator(folder, folder / "operator.h5")
        case_report(folder)
        references.append(folder)
    convergence = compare_operators(
        *references,
        dict(
            energy_eV_per_atom=0.001,
            common_k_frontier_max_eV=0.001,
            potential_rmse_eV=0.005,
            onsite_rmse_eV=0.005,
        ),
    )
    save_json(root / "convergence.json", convergence)
    if not convergence["passed"]:
        raise RuntimeError(
            "Primitive-cell sampling check failed; inspect convergence.json"
        )
    fine = references[-1]
    control = root / "replay_control"
    prepare_replay(fine / "operator.h5", control)
    run(control, executable, launcher=launcher, timeout=3600)
    replay = compare(fine, control, tolerance_eV=1e-4)
    save_json(root / "replay_check.json", replay)
    if not replay["passed"]:
        raise RuntimeError("Native operator replay failed; inspect replay_check.json")
    spectra = []
    for name, source, replacement in (
        ("pbe_bands", fine, {}),
        ("mapho_bands", atomic, dict(potential=potential, d=d)),
    ):
        folder = root / name
        prepare_replay(
            source / "operator.h5", folder, kpoints=path["kpoints"], **replacement
        )
        run(folder, executable, launcher=launcher, timeout=3600)
        data = read_bands(folder)
        if not np.allclose(data["kpoints"], path["kpoints"], atol=1e-7, rtol=0):
            raise ValueError("Calculated band path differs from the requested path")
        spectra.append(data["eigenvalues_eV"])
    metrics = write_results(root, path, *spectra, args.mode)
    save_json(
        root / "completion.json", dict(complete=True, mode=args.mode, metrics=metrics)
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("local", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--meshes", nargs=2, type=int, default=[14, 16])
    parser.add_argument("--launcher", default=shlex.join(LAUNCHER))
    args = parser.parse_args(argv)
    if not 1 <= args.meshes[0] < args.meshes[1]:
        parser.error("--meshes requires two increasing positive integers")
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
