"""Apply frozen Si16 PAW checkpoints to a two-atom Si band structure."""

from ...source_snapshot import snapshot_package, source_hashes
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from ...vasp.paw import (
    prepare_reference,
    prepare_replay,
    run,
    export_operator,
    read_bands,
    sha,
    save_json,
    write_incar,
)
from .silicon_dataset import density_kpoints, case_report
from .training_validation import LAUNCHER, parallel


def read_json(path):
    return json.loads(Path(path).read_text())


def silicon_path(cell, intervals=40):
    """X-Gamma-L in the declared fcc primitive reciprocal basis."""
    vertices = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0.5, 0.5]])
    points = np.concatenate(
        [
            np.linspace(vertices[0], vertices[1], intervals + 1),
            np.linspace(vertices[1], vertices[2], intervals + 1)[1:],
        ]
    )
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    cartesian = points @ reciprocal
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    return dict(
        kpoints=points.tolist(),
        distance_inv_A=distance.tolist(),
        vertices=vertices.tolist(),
        vertex_indices=[0, intervals, 2 * intervals],
        labels=["X", "Gamma", "L"],
        reciprocal_rows_inv_A=reciprocal.tolist(),
    )


def prepare(root, training):
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Poscar

    root, training = Path(root).resolve(), Path(training).resolve()
    if not read_json(training / "training_audit.json")["passed"]:
        raise ValueError("The training artifact audit must pass")
    tm = read_json(training / "manifest.json")
    dataset = read_json(Path(tm["dataset"]) / "manifest.json")
    source = Path(tm["entries"]["train"][0]["baseline"]).parent
    cell = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    structure = Structure(Lattice(cell), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    atoms = AseAtomsAdaptor.get_atoms(structure)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "predictions").mkdir()
    cases = []
    for label, density in [("scf_14", 500), ("scf_16", 750), ("atomic", None)]:
        folder = root / "references" / label
        kp, policy = density_kpoints(structure, density) if density else (None, None)
        mesh = policy["mesh"] if policy else [1, 1, 1]
        m = prepare_reference(
            folder,
            atoms,
            {"Si": source / "POTCAR"},
            mesh=mesh,
            encut=600.0,
            nbands=8,
            algorithm="Normal",
        )
        Poscar(structure, sort_structure=False).write_file(folder / "POSCAR")
        if kp is not None:
            kp.write_file(folder / "KPOINTS")
        m["settings"].update(NELMIN=16, LWAVE=False, LCHARG=label != "atomic")
        if label == "atomic":
            m["settings"].update(ICHARG=12, NELMDL=0)
        write_incar(folder / "INCAR", m["settings"])
        m.update(
            purpose="primitive transfer: fixed atomic density"
            if label == "atomic"
            else "independent primitive PBE reference",
            kpoint_policy=policy,
            inputs={
                n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
            },
        )
        save_json(folder / "manifest.json", m)
        cases.append(label)
    snapshot = root / "source_snapshot" / "mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    manifest = dict(
        schema="mapho.vasp_primitive_bands.v1",
        training=str(training),
        training_manifest_sha256=sha(training / "manifest.json"),
        checkpoints={
            mode: sha(training / mode / "model.pt")
            for mode in ("local", "none", "full")
        },
        executable=dataset["executable"],
        executable_sha256=dataset["executable_sha256"],
        lattice_constant_A=5.43,
        natoms=2,
        reference_cases=cases,
        path=silicon_path(cell),
        source_sha256=source_hashes(snapshot),
        retraining=False,
        input_policy="geometry and fixed atomic-density operator only",
        band_reference="independent native ICHARG=11 from 16x16x16 PBE SCF",
        energy_alignment="no fitted alignment; plot shares native reference VBM zero",
    )
    save_json(root / "manifest.json", manifest)
    save_json(
        root / "figure_contract.json",
        dict(
            question="Does the frozen Si16 model reproduce primitive Si bands?",
            claim="Quantify primitive-cell transfer against native PBE, without retraining.",
            archetype="quantitative grid",
            backend="python",
            panels={
                "a": "Full band dispersion: native PBE versus MACE+FNO",
                "b": "Errors of the top valence and bottom conduction levels, with matched controls",
            },
            exports=[
                "PDF",
                "PNG preview for inspection and inline display",
                "CSV source data",
            ],
            physical_width_mm=180,
            minimum_font_pt=7,
            uncertainty="one crystal, one trained seed; no ensemble intervals",
            integrity="all 81 k points and all eight bands retained in source data; common reference VBM shift only",
            risks=[
                "SCF sampling error checked at 14 and 16 per axis",
                "fixed 600 eV cutoff",
                "band gap sampled on X-Gamma-L, not a general Brillouin-zone search",
            ],
        ),
    )
    return manifest


def reference(root, name):
    root = Path(root)
    m = read_json(root / "manifest.json")
    folder = root / "references" / name
    if sha(m["executable"]) != m["executable_sha256"]:
        raise ValueError("Executable changed")
    if (folder / "CHGCAR").exists() or (folder / "WAVECAR").exists():
        raise ValueError("Fresh reference requires no electronic restart")
    run(folder, m["executable"], launcher=LAUNCHER, timeout=3600)
    export_operator(folder, folder / "operator.h5")
    case_report(folder)


def predict_all(root):
    import torch
    from ...vasp.train import load_model, predict

    torch.set_num_threads(1)
    root = Path(root)
    m = read_json(root / "manifest.json")
    training = Path(m["training"])
    baseline = root / "references/atomic/operator.h5"
    entry = dict(baseline=str(baseline), baseline_sha256=sha(baseline))
    if sha(training / "manifest.json") != m["training_manifest_sha256"]:
        raise ValueError("Training contract changed")
    for mode in ("local", "none", "full"):
        if sha(training / mode / "model.pt") != m["checkpoints"][mode]:
            raise ValueError("Checkpoint changed")
        model = load_model(training, mode, "cuda")
        v, d = predict(model, entry, "cuda")
        np.savez_compressed(root / "predictions" / f"{mode}.npz", potential=v, d=d)
        del model
    # Read the electronic labels only after every prediction is saved.
    with h5py.File(root / "references/scf_16/operator.h5") as f:
        vref, dref = f["potential_eV"][:], f["paw/d"][:]
    metrics = {}
    for mode in ("local", "none", "full"):
        with np.load(root / "predictions" / f"{mode}.npz") as f:
            if f["potential"].shape != vref.shape:
                raise ValueError("Native FFT grids differ")
            metrics[mode] = dict(
                potential_rmse_eV=float(np.mean((f["potential"] - vref) ** 2) ** 0.5),
                onsite_rmse_eV=float(np.mean((f["d"] - dref) ** 2) ** 0.5),
            )
    save_json(root / "field_metrics.json", metrics)


def bands(root, mode):
    root = Path(root)
    m = read_json(root / "manifest.json")
    folder = root / "bands" / mode
    native = mode.startswith("native_")
    source = (
        root
        / "references"
        / (
            "scf_" + mode.split("_")[1]
            if native
            else "scf_16"
            if mode == "replay"
            else "atomic"
        )
    )
    kwargs = {}
    if mode in ("local", "none", "full"):
        with np.load(root / "predictions" / f"{mode}.npz") as f:
            kwargs = dict(potential=f["potential"], d=f["d"])
    if native:
        folder.mkdir(parents=True, exist_ok=False)
        for name in ("POSCAR", "POTCAR", "CHGCAR"):
            shutil.copyfile(source / name, folder / name)
        settings = read_json(source / "manifest.json")["settings"].copy()
        settings.update(ICHARG=11, ISTART=0, NELMDL=0, LWAVE=False, LCHARG=False)
        write_incar(folder / "INCAR", settings)
        points = m["path"]["kpoints"]
        (folder / "KPOINTS").write_text(
            "primitive X-Gamma-L\n"
            + str(len(points))
            + "\nReciprocal\n"
            + "".join(" ".join(f"{v:.16g}" for v in row) + " 1\n" for row in points)
        )
        save_json(
            folder / "manifest.json",
            dict(
                unpatched_control=True,
                settings=settings,
                inputs={
                    n: sha(folder / n)
                    for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
                },
            ),
        )
    else:
        prepare_replay(
            source / "operator.h5", folder, kpoints=m["path"]["kpoints"], **kwargs
        )
    run(folder, m["executable"], launcher=LAUNCHER, timeout=3600)


def analyze(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    modes = ("native_14", "native_16", "replay", "atomic", "local", "none", "full")
    spectra = {}
    for mode in modes:
        data = read_bands(root / "bands" / mode)
        if not np.allclose(data["kpoints"], m["path"]["kpoints"], atol=1e-7, rtol=0):
            raise ValueError("Band paths differ")
        spectra[mode] = data["eigenvalues_eV"]
        if spectra[mode].shape != (81, 8):
            raise ValueError("Unexpected band count")
    ref = spectra["native_16"]
    metrics = {}
    for mode, e in spectra.items():
        delta = e - ref
        vbm_index = int(np.argmax(e[:, 3]))
        cbm_index = int(np.argmin(e[:, 4]))
        metrics[mode] = dict(
            all_band_mae_eV=float(abs(delta).mean()),
            frontier_mae_eV=float(abs(delta[:, 2:6]).mean()),
            frontier_max_eV=float(abs(delta[:, 2:6]).max()),
            vbm_eV=float(e[vbm_index, 3]),
            cbm_eV=float(e[cbm_index, 4]),
            sampled_gap_eV=float(e[cbm_index, 4] - e[vbm_index, 3]),
            vbm_index=vbm_index,
            cbm_index=cbm_index,
        )
    with (
        h5py.File(root / "references/scf_14/operator.h5") as a,
        h5py.File(root / "references/scf_16/operator.h5") as b,
    ):
        conv = dict(
            potential_rmse_eV=float(
                np.mean((a["potential_eV"][:] - b["potential_eV"][:]) ** 2) ** 0.5
            ),
            onsite_rmse_eV=float(np.mean((a["paw/d"][:] - b["paw/d"][:]) ** 2) ** 0.5),
            frontier_max_eV=metrics["native_14"]["frontier_max_eV"],
        )
    conv["energy_eV_per_atom"] = (
        abs(
            read_json(root / "references/scf_14/report.json")["energy_eV"]
            - read_json(root / "references/scf_16/report.json")["energy_eV"]
        )
        / 2
    )
    conv["passed"] = bool(
        conv["frontier_max_eV"] < 0.001
        and conv["potential_rmse_eV"] < 0.001
        and conv["onsite_rmse_eV"] < 0.001
        and conv["energy_eV_per_atom"] < 0.001
    )
    replay_passed = metrics["replay"]["frontier_max_eV"] < 0.001
    result = dict(
        complete=conv["passed"] and replay_passed,
        job=os.environ.get("SLURM_JOB_ID"),
        sampling_convergence=conv,
        reference_replay_passed=replay_passed,
        metrics=metrics,
        fields=read_json(root / "field_metrics.json"),
        path=m["path"],
        frontier_definition="bands 3-6, two valence and two conduction, all 81 path points",
        retraining=False,
        eigenvalues_eV={k: v.tolist() for k, v in spectra.items()},
        source_hashes={
            str(p.relative_to(root)): sha(p)
            for p in (root / "bands").glob("*/EIGENVAL")
        },
        checkpoint_hashes=m["checkpoints"],
    )
    save_json(root / "results.json", result)
    if not result["complete"]:
        raise RuntimeError(
            "Native convergence/replay gate failed; inspect results.json"
        )
    return result


def execute(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    def command(stage, name):
        return (
            stage + "_" + name,
            [
                sys.executable,
                "-u",
                "-m",
                "mapho.commands.vasp.predict",
                "silicon-bands",
                stage,
                str(root),
                name,
            ],
        )

    parallel(
        root, [command("reference", name) for name in m["reference_cases"]], devices
    )
    predict_all(root)
    parallel(
        root,
        [
            command("bands", name)
            for name in (
                "native_14",
                "native_16",
                "replay",
                "atomic",
                "local",
                "none",
                "full",
            )
        ],
        devices,
    )
    analyze(root)
