"""Gated PBE PAW pilots for binary bulk and bilayer structures.

Geometry/sampling generation follows the Si pilot. Slabs retain a fixed vacuum
and one k point normal to the layers. Each stage is independently restartable
from certified completed cases; failed native calculations are never reused.
"""

from ..source_snapshot import snapshot_package, source_hashes, source_path
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

from .paw import prepare_reference, export_operator, run, save_json, sha, potcar_info
from .silicon_dataset import density_kpoints, compare_operators, case_report
from .training_validation import LAUNCHER, parallel


def structure(material, seed=None, calibration=False, vacuum=20.0):
    from pymatgen.core import Lattice, Structure

    if material == "lif":
        a = 4.03
        base = Structure(
            Lattice(a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])),
            ["Li", "F"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        base.make_supercell([2, 2, 2])
        minimum = 1.75
    elif material == "mos2":
        a = 3.18
        thickness = 3.19
        separation = 6.15
        c = vacuum + separation + thickness
        lattice = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        xy = np.array(
            [[0, 0], [2 / 3, 1 / 3], [2 / 3, 1 / 3], [2 / 3, 1 / 3], [0, 0], [0, 0]]
        )
        z = (
            np.array(
                [
                    -separation / 2,
                    -separation / 2 - thickness / 2,
                    -separation / 2 + thickness / 2,
                    separation / 2,
                    separation / 2 - thickness / 2,
                    separation / 2 + thickness / 2,
                ]
            )
            + c / 2
        )
        base = Structure(
            lattice, ["Mo", "S", "S", "Mo", "S", "S"], np.column_stack([xy, z / c])
        )
        base.make_supercell([2, 2, 1])
        minimum = 2.1
    else:
        raise ValueError("Materials: lif, mos2")
    base = base.get_sorted_structure(key=lambda site: site.specie.Z)
    strain = np.zeros((3, 3))
    displacements = np.zeros((len(base), 3))
    layer_delta = 0.0
    if seed is not None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.005, 0.005, (3, 3))
        strain = (raw + raw.T) / 2 + np.eye(3) * rng.uniform(-0.02, 0.02)
        if calibration:
            strain = np.array(
                [[-0.02, 0.005, -0.005], [0.005, -0.02, 0.005], [-0.005, 0.005, -0.02]]
            )
        if material == "mos2":
            strain[2, :] = 0
            strain[:, 2] = 0
            layer_delta = 6.15 * (-0.02 if calibration else rng.uniform(-0.02, 0.02))
        displacements = rng.uniform(-0.04, 0.04, (len(base), 3))
        displacements -= displacements.mean(0)
    cell = base.lattice.matrix @ (np.eye(3) + strain)
    positions = base.frac_coords @ cell + displacements
    if material == "mos2":
        positions[:, 2] += (
            np.sign(base.cart_coords[:, 2] - cell[2, 2] / 2) * layer_delta / 2
        )
    result = Structure(
        cell, base.species, positions, coords_are_cartesian=True, to_unit_cell=True
    )
    distances = result.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    if distances.min() < minimum:
        raise ValueError("Unexpected short separation")
    info = dict(
        material=material,
        seed=seed,
        strain=strain.tolist(),
        displacements_A=displacements.tolist(),
        minimum_distance_A=float(distances.min()),
        layer_separation_delta_A=layer_delta,
        ideal_geometry_not_relaxed=True,
    )
    if material == "mos2":
        info.update(
            stacking="2H antiparallel layers, Mo over opposite-layer S",
            nominal_vacuum_A=vacuum,
            actual_outer_surface_gap_A=float(cell[2, 2] - np.ptp(positions[:, 2])),
            ideal_layer_Mo_separation_A=6.15,
        )
    return result, info


def kpoints(geometry, density, slab=False):
    from pymatgen.io.vasp.inputs import Kpoints

    kp, meta = density_kpoints(geometry, density)
    if slab:
        raw = meta["mesh"].copy()
        mesh = [*raw[:2], 1]
        kp = Kpoints.gamma_automatic(mesh)
        meta.update(
            mesh=mesh,
            raw_3d_proposal=raw,
            normal_sampling="Gamma only along slab normal",
            realized_points_per_reciprocal_volume=float(
                np.prod(mesh) / geometry.lattice.reciprocal_lattice.volume
            ),
        )
    return kp, meta


def write_case(folder, geometry, info, potcars, density, role, nbands):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Incar, Poscar

    folder = Path(folder)
    kp, policy = kpoints(geometry, density, info["material"] == "mos2")
    manifest = prepare_reference(
        folder,
        AseAtomsAdaptor.get_atoms(geometry),
        potcars,
        encut=600.0,
        mesh=policy["mesh"],
        nbands=nbands,
        algorithm="Normal",
    )
    Poscar(geometry, sort_structure=False).write_file(folder / "POSCAR")
    manifest["settings"].update(NELMIN=12, LWAVE=False, LCHARG=True)
    Incar(manifest["settings"]).write_file(folder / "INCAR")
    kp.write_file(folder / "KPOINTS")
    manifest.update(
        purpose=role,
        geometry=info,
        kpoint_policy=policy,
        inputs={n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")},
    )
    save_json(folder / "manifest.json", manifest)
    return dict(
        directory=str(folder.resolve()), role=role, geometry=info, kpoints=policy
    )


def prepare(root, material, potcar_root, executable, count=24, seed=29, densities=None):
    densities = tuple(
        densities
        or ((200.0, 300.0, 500.0) if material == "lif" else (500.0, 750.0, 1000.0))
    )
    root = Path(root).resolve()
    potcar_root = Path(potcar_root).resolve()
    choices = {"lif": {"Li": "Li_sv", "F": "F"}, "mos2": {"S": "S", "Mo": "Mo_pv"}}[
        material
    ]
    potcars = {
        symbol: str(potcar_root / name / "POTCAR") for symbol, name in choices.items()
    }
    infos = {s: potcar_info(p, s) for s, p in potcars.items()}
    pristine, _ = structure(material)
    nelect = sum(infos[s.symbol]["zval"] for s in pristine.species)
    nbands = int(np.ceil((nelect / 2 + 16) / 8) * 8)
    if (
        count < 8
        or len(densities) < 3
        or any(b <= a for a, b in zip(densities, densities[1:]))
    ):
        raise ValueError(
            "At least eight samples and three increasing densities required"
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    samples = []
    calibration = []
    for i in range(count):
        geometry, info = structure(material, seed * 10000 + i)
        samples.append(
            write_case(
                root / "samples" / f"sample_{i:03d}",
                geometry,
                info,
                potcars,
                densities[-1],
                "production reference; k convergence pending",
                nbands,
            )
        )
    for name, geometry_seed in [("pristine", None), ("distorted", 19073)]:
        geometry, info = structure(
            material, geometry_seed, calibration=geometry_seed is not None
        )
        meshes = []
        for density in densities:
            row = write_case(
                root / "convergence" / name / f"density_{density:g}",
                geometry,
                info,
                potcars,
                density,
                "k-point calibration; excluded from splits",
                nbands,
            )
            row["geometry_name"] = name
            calibration.append(row)
            meshes.append(tuple(row["kpoints"]["mesh"]))
        if len(set(meshes)) != len(meshes):
            raise ValueError("Densities must produce distinct meshes")
    order = np.random.default_rng(seed).permutation(count)
    nval = max(1, count // 6)
    ntrain = count - 2 * nval
    splits = {
        key: [f"samples/sample_{i:03d}/operator.h5" for i in indices]
        for key, indices in [
            ("train", order[:ntrain]),
            ("validation", order[ntrain : ntrain + nval]),
            ("test", order[ntrain + nval :]),
        ]
    }
    snapshot = root / "source_snapshot/mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    manifest = dict(
        schema="mapho.vasp_material_dataset.v1",
        material=material,
        count=count,
        seed=seed,
        natoms=len(pristine),
        nelect=nelect,
        nbands=nbands,
        encut_eV=600.0,
        potcars=potcars,
        potentials=infos,
        executable=str(Path(executable).resolve()),
        executable_sha256=sha(executable),
        samples=samples,
        calibration=calibration,
        splits=splits,
        production_kppvol=densities[-1],
        calibration_kppvol=list(densities),
        source_sha256=source_hashes(snapshot),
        thresholds=dict(
            energy_eV_per_atom=0.001,
            common_k_frontier_max_eV=0.001,
            potential_rmse_eV=0.005,
            onsite_rmse_eV=0.005,
        ),
        physics="Nonmagnetic PBE, no SOC or empirical dispersion, fixed ideal reference geometry plus small perturbations",
        limitations=[
            "Fixed 600 eV cutoff; not a cutoff convergence claim.",
            "Small cells and one training seed are pilot comparisons.",
            "Bilayer has 20 A nominal vacuum; vacuum/grid checks remain separate.",
        ],
    )
    save_json(root / "manifest.json", manifest)
    return manifest


def read_json(path):
    return json.loads(Path(path).read_text())


def convergence(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    checks = []
    for name in dict.fromkeys(r["geometry_name"] for r in m["calibration"]):
        rows = [r for r in m["calibration"] if r["geometry_name"] == name]
        for a, b in zip(rows, rows[1:]):
            check = dict(
                geometry=name, coarse=a["kpoints"], fine=b["kpoints"], passed=False
            )
            try:
                check.update(
                    compare_operators(a["directory"], b["directory"], m["thresholds"])
                )
            except (ValueError, OSError, KeyError) as exc:
                check["error"] = str(exc)
            checks.append(check)
    passed = bool(checks) and all(c["passed"] for c in checks)
    result = dict(
        passed=passed,
        checks=checks,
        selected_kppvol=m["production_kppvol"] if passed else None,
        thresholds=m["thresholds"],
        gauge="Native V and D, no fitted energy shifts",
    )
    save_json(root / "convergence.json", result)
    return result


def worker(root, phase, index):
    root = Path(root)
    m = read_json(root / "manifest.json")
    folder = Path(m[phase][index]["directory"])
    if sha(m["executable"]) != m["executable_sha256"]:
        raise ValueError("Executable changed")
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise ValueError("Worker snapshot changed")
    if phase == "samples" and not read_json(root / "convergence.json")["passed"]:
        raise ValueError("Sampling gate failed")
    run(folder, m["executable"], launcher=LAUNCHER, timeout=6600)
    if "Offloading initialized" not in (folder / "stdout.log").read_text():
        raise ValueError("GPU offload absent")
    export_operator(folder, folder / "operator.h5")
    case_report(folder)


def audit(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    records = []
    if not read_json(root / "convergence.json")["passed"]:
        raise ValueError("Sampling gate failed")
    for row in m["samples"]:
        p = Path(row["directory"])
        r = read_json(p / "report.json")
        runinfo = read_json(p / "run.json")
        if (
            not r["complete"]
            or runinfo["status"] != "complete"
            or sha(p / "operator.h5") != r["operator_sha256"]
        ):
            raise ValueError("Incomplete or changed reference")
        with h5py.File(p / "operator.h5") as f:
            for key in ("potential_eV", "paw/d", "paw/q", "bands/eigenvalues_eV"):
                if not np.isfinite(f[key][:]).all():
                    raise ValueError("Non-finite exported data")
            if len(f["atomic_numbers"]) != m["natoms"]:
                raise ValueError("Atom count changed")
            records.append(
                dict(
                    name=p.name,
                    sha256=r["operator_sha256"],
                    grid=r["grid"],
                    channels=json.loads(f["paw"].attrs["channels_json"]),
                    sampled_gap_eV=r["sampled_gap_eV"],
                )
            )
    result = dict(
        passed=True,
        count=len(records),
        records=records,
        dataset_manifest_sha256=sha(root / "manifest.json"),
    )
    save_json(root / "data_audit.json", result)
    save_json(
        root / "completion.json",
        dict(
            complete=True,
            count=len(records),
            job=os.environ.get("SLURM_JOB_ID"),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return result


def execute(root, phase):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    if phase == "samples" and not convergence(root)["passed"]:
        raise RuntimeError("Sampling failed; production not started")
    commands = []
    for i, row in enumerate(m[phase]):
        folder = Path(row["directory"])
        if (folder / "report.json").exists():
            r = read_json(folder / "report.json")
            if r["complete"] and sha(folder / "operator.h5") == r["operator_sha256"]:
                continue
        if (folder / "run.json").exists():
            raise RuntimeError(f"Uncertified previous calculation: {folder}")
        commands.append(
            (
                f"{phase}_{i:03d}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    str(root),
                    phase,
                    str(i),
                ],
            )
        )
    parallel(root, commands, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    result = convergence(root) if phase == "calibration" else audit(root)
    if not result["passed"]:
        raise RuntimeError("Convergence gate failed; inspect convergence.json")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root")
    q.add_argument("material", choices=["lif", "mos2"])
    q.add_argument("--potcar-root", required=True)
    q.add_argument("--executable", required=True)
    q = sub.add_parser("execute")
    q.add_argument("root")
    q.add_argument("phase", choices=["calibration", "samples"])
    q = sub.add_parser("worker")
    q.add_argument("root")
    q.add_argument("phase")
    q.add_argument("index", type=int)
    q = sub.add_parser("audit")
    q.add_argument("root")
    a = p.parse_args()
    if a.command == "prepare":
        m = prepare(a.root, a.material, a.potcar_root, a.executable)
        print(
            json.dumps(
                {
                    k: m[k]
                    for k in (
                        "material",
                        "count",
                        "natoms",
                        "nbands",
                        "production_kppvol",
                    )
                }
            )
        )
    elif a.command == "worker":
        worker(a.root, a.phase, a.index)
    elif a.command == "execute":
        execute(a.root, a.phase)
    else:
        audit(a.root)
