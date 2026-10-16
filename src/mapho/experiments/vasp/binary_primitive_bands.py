"""Frozen LiF16 and MoS2-bilayer24 checkpoints on their pristine primitive cells."""

from ...source_snapshot import snapshot_package, source_hashes
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from .material_dataset import structure, kpoints
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
from .silicon_dataset import case_report
from .training_validation import LAUNCHER, parallel


def read_json(path):
    return json.loads(Path(path).read_text())


def primitive(material):
    """Undo only the declared repetitions; retain the generator's lattice basis."""
    from pymatgen.core import Structure

    parent, info = structure(material)
    repeats = np.array([2, 2, 2] if material == "lif" else [2, 2, 1])
    cell = parent.lattice.matrix / repeats[:, None]
    species, coords = [], []
    for site in parent:
        frac = np.mod(site.frac_coords * repeats, 1.0)
        if any(
            s == site.specie and np.max(abs((frac - f) - np.rint(frac - f))) < 1e-8
            for s, f in zip(species, coords)
        ):
            continue
        species.append(site.specie)
        coords.append(frac)
    result = Structure(cell, species, coords)
    expected = 2 if material == "lif" else 6
    if len(result) != expected:
        raise ValueError("Primitive reduction produced the wrong atom count")
    rebuilt = result.copy()
    rebuilt.make_supercell(repeats)
    if not np.allclose(rebuilt.lattice.matrix, parent.lattice.matrix):
        raise ValueError("Reconstructed parent cell differs")
    for site in parent:
        if not any(
            other.specie == site.specie and site.distance(other) < 1e-7
            for other in rebuilt
        ):
            raise ValueError(
                "Primitive structure does not reconstruct the training parent"
            )
    return result, dict(
        parent=info,
        repetitions_undone=repeats.tolist(),
        relaxed=False,
        provenance="Unstrained parent from vasp_material_dataset.structure; no experimental-source claim",
    )


def band_path(material, cell):
    if material == "lif":
        vertices = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0.5, 0.5]])
        labels, intervals = ["X", "Gamma", "L"], 40
    else:
        vertices = np.array([[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]])
        labels, intervals = ["Gamma", "M", "K", "Gamma"], 24
    points = np.concatenate(
        [
            np.linspace(a, b, intervals + 1)[int(i > 0) :]
            for i, (a, b) in enumerate(zip(vertices[:-1], vertices[1:]))
        ]
    )
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    distance = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(points @ reciprocal, axis=0), axis=1))
    ]
    assert np.all(np.diff(distance) > 0)
    return dict(
        kpoints=points.tolist(),
        distance_inv_A=distance.tolist(),
        vertices=vertices.tolist(),
        labels=labels,
        vertex_indices=(np.arange(len(labels)) * intervals).tolist(),
        reciprocal_rows_inv_A=reciprocal.tolist(),
    )


def prepare(root):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Poscar

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    runs = root.parent
    specs = {
        "lif": dict(
            dataset=runs / "vasp_lif16_v1",
            training=runs / "vasp_lif16_train_v1/silu",
            repaired=False,
            densities=[500.0, 750.0],
            nbands=16,
        ),
        "mos2": dict(
            dataset=runs / "vasp_mos2_bilayer24_v1",
            training=runs / "vasp_mos2_paw_conditioned_v3",
            repaired=True,
            densities=[1000.0, 1500.0],
            nbands=40,
        ),
    }
    for material, spec in specs.items():
        dataset = read_json(spec["dataset"] / "manifest.json")
        training = spec["training"]
        geometry, info = primitive(material)
        spec.update(
            training_manifest_sha256=sha(training / "manifest.json"),
            checkpoints={
                mode: sha(
                    training
                    / (("silu_" + mode) if spec["repaired"] else mode)
                    / "model.pt"
                )
                for mode in ("none", "full")
            },
            executable=dataset["executable"],
            executable_sha256=dataset["executable_sha256"],
            natoms=len(geometry),
            nocc=5 if material == "lif" else 24,
            cell_A=geometry.lattice.matrix.tolist(),
            geometry=info,
            path=band_path(material, geometry.lattice.matrix),
        )
        (root / material / "predictions").mkdir(parents=True)
        for label, density in zip(
            ("scf_low", "scf_high", "atomic"), (*spec["densities"], None)
        ):
            folder = root / material / "references" / label
            kp, policy = (
                kpoints(geometry, density, material == "mos2")
                if density
                else (None, None)
            )
            mesh = policy["mesh"] if policy else [1, 1, 1]
            m = prepare_reference(
                folder,
                AseAtomsAdaptor.get_atoms(geometry),
                dataset["potcars"],
                mesh=mesh,
                encut=600.0,
                nbands=spec["nbands"],
                algorithm="Normal",
            )
            Poscar(geometry, sort_structure=False).write_file(folder / "POSCAR")
            if kp is not None:
                kp.write_file(folder / "KPOINTS")
            m["settings"].update(
                NELMIN=16,
                LWAVE=False,
                LCHARG=label != "atomic",
                ISYM=-1 if label == "atomic" else 2,
            )
            if label == "atomic":
                m["settings"].update(ICHARG=12, NELMDL=0)
            write_incar(folder / "INCAR", m["settings"])
            m.update(
                purpose="Primitive-cell frozen-model transfer",
                geometry=info,
                kpoint_policy=policy,
                inputs={
                    n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
                },
            )
            save_json(folder / "manifest.json", m)
        spec["dataset"], spec["training"] = str(spec["dataset"]), str(training)
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    manifest = dict(
        schema="mapho.vasp_binary_primitive_bands.v1",
        materials=specs,
        source_sha256=source_hashes(snapshot),
        retraining=False,
        model_selection="SiLU FNO plus matched pointwise control, selected before primitive spectra",
        band_reference="Native ICHARG=11 from independent denser PBE SCF",
        energy_alignment="Common reference VBM zero; no fitted or method-specific shifts",
    )
    save_json(root / "manifest.json", manifest)
    save_json(
        root / "figure_contract.json",
        dict(
            question="How accurately do the frozen supercell models transfer to pristine primitive-cell bands?",
            claim="Report measured dispersion agreement and errors for LiF and the MoS2 bilayer without retraining.",
            archetype="quantitative comparison",
            backend="python",
            panels={
                "a": "LiF reference versus FNO dispersion",
                "b": "MoS2 reference versus FNO dispersion",
                "c": "LiF frontier errors with pointwise control",
                "d": "MoS2 frontier errors with pointwise control",
            },
            exports=[
                "PDF",
                "PNG for requested inline display and visual QA",
                "CSV source energies",
            ],
            physical_width_mm=183,
            minimum_font_pt=7,
            uncertainty="One ideal structure and one fitted seed per material; no ensemble uncertainty",
            integrity="All k points and bands in source data; common reference VBM shift; fixed display windows disclosed",
            limits=[
                "Not relaxed structures",
                "Nonmagnetic PBE, no SOC or dispersion",
                "Fixed 600 eV cutoff",
                "Path-sampled gaps, not full Brillouin-zone minimization",
                "No retraining on primitive reference labels",
            ],
        ),
    )
    return manifest


def reference(root, material, name):
    root = Path(root)
    spec = read_json(root / "manifest.json")["materials"][material]
    folder = root / material / "references" / name
    if sha(spec["executable"]) != spec["executable_sha256"]:
        raise ValueError("Executable changed")
    if (folder / "CHGCAR").exists() or (folder / "WAVECAR").exists():
        raise ValueError("Fresh reference required")
    run(folder, spec["executable"], launcher=LAUNCHER, timeout=5400)
    export_operator(folder, folder / "operator.h5")
    case_report(folder)


def predict_material(root, material):
    import torch
    from ...vasp.train import load_model, predict
    from ...vasp.checkpoints import load_repaired_model

    torch.set_num_threads(1)
    root = Path(root)
    spec = read_json(root / "manifest.json")["materials"][material]
    training = Path(spec["training"])
    baseline = root / material / "references/atomic/operator.h5"
    if sha(training / "manifest.json") != spec["training_manifest_sha256"]:
        raise ValueError("Training contract changed")
    for mode in ("none", "full"):
        label = "silu_" + mode if spec["repaired"] else mode
        if sha(training / label / "model.pt") != spec["checkpoints"][mode]:
            raise ValueError("Checkpoint changed")
        model = (load_repaired_model if spec["repaired"] else load_model)(
            training, label, "cuda"
        )
        v, d = predict(
            model, dict(baseline=str(baseline), baseline_sha256=sha(baseline)), "cuda"
        )
        if not np.isfinite(v).all() or not np.isfinite(d).all():
            raise ValueError("Nonfinite prediction")
        np.savez_compressed(
            root / material / "predictions" / f"{mode}.npz", potential=v, d=d
        )
        del model
        torch.cuda.empty_cache()
    with h5py.File(root / material / "references/scf_high/operator.h5") as f:
        vref, dref = f["potential_eV"][:], f["paw/d"][:]
    metrics = {}
    for mode in ("none", "full"):
        with np.load(root / material / "predictions" / f"{mode}.npz") as f:
            if f["potential"].shape != vref.shape:
                raise ValueError("Native grids differ")
            metrics[mode] = dict(
                potential_rmse_eV=float(np.mean((f["potential"] - vref) ** 2) ** 0.5),
                onsite_rmse_eV=float(np.mean((f["d"] - dref) ** 2) ** 0.5),
            )
    save_json(root / material / "field_metrics.json", metrics)


def bands(root, material, mode):
    root = Path(root)
    spec = read_json(root / "manifest.json")["materials"][material]
    base = root / material
    folder = base / "bands" / mode
    points = spec["path"]["kpoints"]
    if mode in ("native_low", "native_high"):
        source = base / "references" / ("scf_" + mode.split("_")[1])
        folder.mkdir(parents=True, exist_ok=False)
        for name in ("POSCAR", "POTCAR", "CHGCAR"):
            shutil.copyfile(source / name, folder / name)
        settings = read_json(source / "manifest.json")["settings"].copy()
        settings.update(
            ICHARG=11, ISTART=0, NELMDL=0, LWAVE=False, LCHARG=False, ISYM=-1, NELMIN=32
        )
        write_incar(folder / "INCAR", settings)
        (folder / "KPOINTS").write_text(
            "Primitive band path\n"
            + str(len(points))
            + "\nReciprocal\n"
            + "".join(" ".join(f"{v:.16g}" for v in row) + " 1\n" for row in points)
        )
        (folder / "MLHAM_EXPORT").touch()
        m = read_json(source / "manifest.json")
        m.update(
            settings=settings,
            purpose="Native frozen-density bands with operator export",
            inputs={
                n: sha(folder / n)
                for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "CHGCAR")
            },
        )
    else:
        source = (
            base
            / "references"
            / ("scf_high" if mode == "scf_replay" else "atomic")
            / "operator.h5"
        )
        kwargs = {}
        if mode == "native_replay":
            source = base / "bands/native_high/operator.h5"
        if mode in ("none", "full"):
            with np.load(base / "predictions" / f"{mode}.npz") as f:
                kwargs = dict(potential=f["potential"], d=f["d"])
        m = prepare_replay(source, folder, kpoints=points, **kwargs)
        m["settings"].update(NELMIN=32, EDIFF=1e-9, ISYM=-1)
        write_incar(folder / "INCAR", m["settings"])
        m["inputs"]["INCAR"] = sha(folder / "INCAR")
    save_json(folder / "manifest.json", m)
    run(folder, spec["executable"], launcher=LAUNCHER, timeout=5400)
    if mode.startswith("native_") and mode != "native_replay":
        export_operator(folder, folder / "operator.h5")


def analyze(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    results = {}
    for material, spec in manifest["materials"].items():
        base = root / material
        spectra = {}
        nocc = spec["nocc"]
        window = slice(nocc - 2, nocc + 2)
        for mode in (
            "native_low",
            "native_high",
            "scf_replay",
            "none",
            "full",
            "native_replay",
        ):
            folder = base / "bands" / mode
            if read_json(folder / "run.json")["status"] != "complete":
                raise ValueError("Uncertified band solve")
            data = read_bands(folder)
            if not np.allclose(
                data["kpoints"], spec["path"]["kpoints"], atol=1e-7, rtol=0
            ):
                raise ValueError("Path mismatch")
            e = data["eigenvalues_eV"]
            if (
                e.shape != (len(spec["path"]["kpoints"]), spec["nbands"])
                or not np.isfinite(e).all()
            ):
                raise ValueError("Invalid bands")
            spectra[mode] = e
        ref = spectra["native_high"]
        metrics = {}
        for mode, e in spectra.items():
            delta = e - ref
            vi = int(np.argmax(e[:, nocc - 1]))
            ci = int(np.argmin(e[:, nocc]))
            metrics[mode] = dict(
                all_band_mae_eV=float(abs(delta).mean()),
                all_band_max_eV=float(abs(delta).max()),
                frontier_mae_eV=float(abs(delta[:, window]).mean()),
                frontier_max_eV=float(abs(delta[:, window]).max()),
                vbm_eV=float(e[vi, nocc - 1]),
                cbm_eV=float(e[ci, nocc]),
                sampled_gap_eV=float(e[ci, nocc] - e[vi, nocc - 1]),
                vbm_index=vi,
                cbm_index=ci,
            )
        refs = base / "references"
        with (
            h5py.File(refs / "scf_low/operator.h5") as a,
            h5py.File(refs / "scf_high/operator.h5") as b,
        ):
            convergence = dict(
                potential_rmse_eV=float(
                    np.mean((a["potential_eV"][:] - b["potential_eV"][:]) ** 2) ** 0.5
                ),
                onsite_rmse_eV=float(
                    np.mean((a["paw/d"][:] - b["paw/d"][:]) ** 2) ** 0.5
                ),
                frontier_max_eV=metrics["native_low"]["frontier_max_eV"],
            )
        convergence["energy_eV_per_atom"] = (
            abs(
                read_json(refs / "scf_low/report.json")["energy_eV"]
                - read_json(refs / "scf_high/report.json")["energy_eV"]
            )
            / spec["natoms"]
        )
        convergence["passed"] = all(
            convergence[k] < threshold
            for k, threshold in [
                ("potential_rmse_eV", 0.005),
                ("onsite_rmse_eV", 0.005),
                ("frontier_max_eV", 0.001),
                ("energy_eV_per_atom", 0.001),
            ]
        )
        bridge = dict(
            frontier_max_eV=metrics["native_replay"]["frontier_max_eV"],
            all_band_max_eV=metrics["native_replay"]["all_band_max_eV"],
            passed=metrics["native_replay"]["all_band_max_eV"] < 0.001,
        )
        results[material] = dict(
            specification=spec,
            metrics=metrics,
            sampling_convergence=convergence,
            native_export_replay=bridge,
            fields=read_json(base / "field_metrics.json"),
            eigenvalues_eV={k: v.tolist() for k, v in spectra.items()},
            source_sha256={
                str(p.relative_to(root)): sha(p)
                for p in (base / "bands").glob("*/EIGENVAL")
            },
        )
    out = dict(
        complete=all(
            r["sampling_convergence"]["passed"] and r["native_export_replay"]["passed"]
            for r in results.values()
        ),
        job=os.environ.get("SLURM_JOB_ID"),
        materials=results,
        frontier_definition="Two highest valence plus two lowest conduction bands, every path point",
        reconstruction_difference="SCF export versus CHGCAR reconstruction is reported separately, not confused with same-operator replay",
    )
    save_json(root / "results.json", out)
    if not out["complete"]:
        raise RuntimeError("Reference certification failed; inspect results.json")
    return out


def execute(root):
    root = Path(root).resolve()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    def cmd(stage, material, name=None):
        args = [
            sys.executable,
            "-u",
            "-m",
            "mapho.commands.vasp.predict",
            "binary-bands",
            stage,
            str(root),
            material,
        ]
        if name is not None:
            args.append(name)
        return ("_".join(x for x in (stage, material, name) if x), args)

    parallel(
        root,
        [
            cmd("reference", s, n)
            for s in ("lif", "mos2")
            for n in ("scf_low", "scf_high", "atomic")
        ],
        devices,
    )
    parallel(root, [cmd("predict", s) for s in ("lif", "mos2")], devices)
    parallel(
        root,
        [
            cmd("bands", s, n)
            for s in ("lif", "mos2")
            for n in ("native_low", "native_high", "scf_replay", "full", "none")
        ],
        devices,
    )
    parallel(root, [cmd("bands", s, "native_replay") for s in ("lif", "mos2")], devices)
    analyze(root)
