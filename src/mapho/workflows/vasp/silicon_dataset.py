"""Pymatgen Si16 inputs and gated, concurrent native VASP PAW references.

One common reciprocal-volume density is used for every production structure.
Calibration structures are separate from the structure-level dataset splits.
Only reference data are generated here; no PAW learning model is trained.
"""

from ...source_snapshot import snapshot_package, source_hashes, source_path
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from ...vasp.paw import prepare_reference, run, export_operator, save_json, sha


def silicon_structure(seed=None, calibration=False):
    from pymatgen.core import Lattice, Structure

    a = 5.43
    primitive = Structure(
        Lattice(a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    primitive.make_supercell([2, 2, 2])
    strain = np.zeros((3, 3))
    displacements = np.zeros((16, 3))
    if seed is not None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.005, 0.005, (3, 3))
        strain = (raw + raw.T) / 2 + np.eye(3) * rng.uniform(-0.02, 0.02)
        if calibration:
            strain = np.array(
                [[-0.02, 0.005, -0.005], [0.005, -0.02, 0.005], [-0.005, 0.005, -0.02]]
            )
        displacements = rng.uniform(-0.04, 0.04, (16, 3))
        displacements -= displacements.mean(axis=0)
    lattice = primitive.lattice.matrix @ (np.eye(3) + strain)
    positions = primitive.frac_coords @ lattice + displacements
    structure = Structure(
        lattice, ["Si"] * 16, positions, coords_are_cartesian=True, to_unit_cell=True
    )
    distances = structure.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 2.1:
        raise ValueError("Unexpectedly short Si-Si separation")
    return structure, dict(
        seed=seed,
        strain=strain.tolist(),
        displacements_A=displacements.tolist(),
        minimum_distance_A=float(distances.min()),
    )


def density_kpoints(structure, density):
    from pymatgen.io.vasp.inputs import Kpoints

    if not np.isfinite(density) or density <= 0:
        raise ValueError("Reciprocal-volume density must be finite and positive")
    kp = Kpoints.automatic_density_by_vol(structure, density, force_gamma=True)
    mesh = list(map(int, kp.kpts[0]))
    metadata = dict(
        method="pymatgen.Kpoints.automatic_density_by_vol",
        requested_kppvol=float(density),
        mesh=mesh,
        centering="Gamma",
        reciprocal_volume_inv_A3=structure.lattice.reciprocal_lattice.volume,
        realized_points_per_reciprocal_volume=float(
            np.prod(mesh) / structure.lattice.reciprocal_lattice.volume
        ),
        pymatgen_version=version("pymatgen"),
    )
    return kp, metadata


def write_reference(folder, structure, info, potcar, density, role):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Incar, Poscar

    folder = Path(folder)
    kp, proposal = density_kpoints(structure, density)
    manifest = prepare_reference(
        folder,
        AseAtomsAdaptor.get_atoms(structure),
        {"Si": potcar},
        encut=600.0,
        mesh=proposal["mesh"],
        nbands=48,
        algorithm="Normal",
        ispin=1,
    )
    # Pymatgen writes all geometry/settings/sampling files; preserve the explicit
    # bridge settings rather than importing unrelated input-set defaults.
    Poscar(structure, sort_structure=False).write_file(folder / "POSCAR")
    settings = manifest["settings"]
    settings.update(NELMIN=12, LWAVE=False, LCHARG=True)
    Incar(settings).write_file(folder / "INCAR")
    kp.write_file(folder / "KPOINTS")
    manifest.update(
        purpose=role,
        geometry=info,
        kpoint_policy=proposal,
        inputs={n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")},
    )
    save_json(folder / "manifest.json", manifest)
    return dict(
        directory=str(folder.resolve()), role=role, geometry=info, kpoints=proposal
    )


def prepare(
    root, potcar, executable, count=24, seed=29, densities=(200.0, 300.0, 500.0)
):
    root = Path(root).resolve()
    if (
        count < 8
        or len(densities) < 3
        or any(b <= a for a, b in zip(densities, densities[1:]))
    ):
        raise ValueError(
            "Need at least eight samples and three increasing calibration densities"
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    samples = []
    for i in range(count):
        structure, info = silicon_structure(seed * 10000 + i)
        samples.append(
            write_reference(
                root / "samples" / f"sample_{i:03d}",
                structure,
                info,
                potcar,
                densities[-1],
                "production reference; sampling gate pending",
            )
        )
    calibration = []
    for name, geometry_seed in [("pristine", None), ("distorted", 19073)]:
        structure, info = silicon_structure(
            geometry_seed, calibration=geometry_seed is not None
        )
        meshes = []
        for density in densities:
            row = write_reference(
                root / "convergence" / name / f"density_{density:g}",
                structure,
                info,
                potcar,
                density,
                "k-point calibration; excluded from splits",
            )
            row["geometry_name"] = name
            calibration.append(row)
            meshes.append(tuple(row["kpoints"]["mesh"]))
        if len(set(meshes)) != len(meshes):
            raise ValueError("Calibration densities must produce distinct meshes")
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
    snapshot = root / "source_snapshot" / "mapho"
    snapshot.mkdir(parents=True)
    snapshot_package(snapshot)
    manifest = dict(
        schema="mapho.vasp_silicon_dataset.v1",
        seed=seed,
        count=count,
        natoms=16,
        executable=str(Path(executable).resolve()),
        executable_sha256=sha(executable),
        potcar_sha256=sha(potcar),
        samples=samples,
        calibration=calibration,
        splits=splits,
        production_kppvol=float(densities[-1]),
        calibration_kppvol=list(densities),
        source_sha256=source_hashes(snapshot),
        distribution="Si diamond a=5.43 A; 2x2x2 primitive supercells; independent isotropic strain +/-2%, symmetric strain components +/-0.5%, per-atom displacement components +/-0.04 A before translation removal",
        thresholds=dict(
            energy_eV_per_atom=0.001,
            common_k_frontier_max_eV=0.001,
            potential_rmse_eV=0.005,
            onsite_rmse_eV=0.005,
        ),
        limitations=[
            "K-point convergence at fixed 600 eV; cutoff convergence remains separate.",
            "Coarse-mesh band extrema are diagnostics, not converged fundamental gaps.",
            "Small-cell data pilot, not a test of long-range model superiority.",
            "No trained PAW model or residual atomic-reference convention is supplied.",
        ],
    )
    save_json(root / "manifest.json", manifest)
    return manifest


def case_report(folder):
    folder = Path(folder)
    with h5py.File(folder / "operator.h5") as f:
        bands = f["bands/eigenvalues_eV"][:]
        nocc = int(
            json.loads(f.attrs["metadata_json"])["reference_manifest"]["nelect"] / 2
        )
        shape = list(f["potential_eV"].shape)
    xml = ET.parse(folder / "vasprun.xml").getroot()
    energy = xml.findall("calculation")[-1].find("energy")
    values = {v.attrib["name"]: float(v.text) for v in energy.findall("i")}
    result = dict(
        complete=True,
        operator_sha256=sha(folder / "operator.h5"),
        energy_eV=values["e_0_energy"],
        free_energy_eV=values["e_fr_energy"],
        nocc=nocc,
        sampled_gap_eV=float(bands[:, nocc].min() - bands[:, nocc - 1].max()),
        grid=shape,
        nkpoints=len(bands),
        nbands=bands.shape[1],
        gpu_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if result["sampled_gap_eV"] <= 0:
        raise RuntimeError("Pilot is restricted to insulating Si structures")
    save_json(folder / "report.json", result)
    return result


def run_case(root, phase, index):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if sha(m["executable"]) != m["executable_sha256"]:
        raise RuntimeError("Validated executable changed")
    for name, digest in m["source_sha256"].items():
        if sha(source_path(name)) != digest:
            raise RuntimeError("Worker source differs from study snapshot")
    if phase == "samples":
        gate = json.loads((root / "convergence.json").read_text())
        if not gate["passed"] or gate["selected_kppvol"] != m["production_kppvol"]:
            raise RuntimeError(
                "Sampling gate did not authorize these production inputs"
            )
    folder = Path(m[phase][index]["directory"])
    launcher = [
        "mpirun",
        "--bind-to",
        "none",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,vader",
        "-np",
        "1",
    ]
    run(folder, m["executable"], launcher=launcher, timeout=6600)
    if "Offloading initialized" not in (folder / "stdout.log").read_text():
        raise RuntimeError("GPU offloading was not reported")
    export_operator(folder, folder / "operator.h5")
    return case_report(folder)


def compare_operators(a, b, thresholds):
    a, b = Path(a), Path(b)
    ra, rb = [json.loads((p / "report.json").read_text()) for p in (a, b)]
    if not ra["complete"] or not rb["complete"]:
        raise ValueError("Incomplete reference")
    with h5py.File(a / "operator.h5") as fa, h5py.File(b / "operator.h5") as fb:
        for key in ("cell_A", "positions_A", "atomic_numbers", "paw/types", "paw/q"):
            if fa[key].shape != fb[key].shape or not np.allclose(
                fa[key][:], fb[key][:], atol=1e-10, rtol=0
            ):
                raise ValueError(
                    "Convergence comparison requires identical structure and PAW data"
                )
        va, vb = fa["potential_eV"][:], fb["potential_eV"][:]
        if va.shape != vb.shape:
            raise ValueError("Convergence comparison requires the same FFT grid")
        da, db = fa["paw/d"][:], fb["paw/d"][:]
        channels = json.loads(fa["paw"].attrs["channels_json"])
        active = np.zeros(da.shape, dtype=bool)
        for atom, kind in enumerate(fa["paw/types"][:]):
            size = sum(2 * l + 1 for l in channels[kind])
            active[atom, :size, :size] = True
        ea, eb = fa["bands/eigenvalues_eV"][:], fb["bands/eigenvalues_eV"][:]
        ka, kb = fa["bands/kpoints"][:], fb["bands/kpoints"][:]
        pairs = []
        for i, k in enumerate(ka):
            delta = kb - k
            delta -= np.rint(delta)
            hit = np.flatnonzero(np.linalg.norm(delta, axis=1) < 1e-7)
            if len(hit) == 1:
                pairs.append((i, int(hit[0])))
        if not pairs or not any(
            np.linalg.norm(ka[i] - np.rint(ka[i])) < 1e-7 for i, _ in pairs
        ):
            raise ValueError("Common Gamma point required for frontier convergence")
        nocc = ra["nocc"]
        frontier = slice(nocc - 4, nocc + 4)
        values = dict(
            energy_eV_per_atom=abs(ra["energy_eV"] - rb["energy_eV"])
            / len(fa["atomic_numbers"]),
            common_k_frontier_max_eV=max(
                float(np.abs(ea[i, frontier] - eb[j, frontier]).max()) for i, j in pairs
            ),
            potential_rmse_eV=float(np.sqrt(np.mean((va - vb) ** 2))),
            onsite_rmse_eV=float(np.sqrt(np.mean((da[active] - db[active]) ** 2))),
        )
        return dict(
            **values,
            common_k_count=len(pairs),
            potential_mean_change_eV=float((vb - va).mean()),
            sampled_gap_change_eV=rb["sampled_gap_eV"] - ra["sampled_gap_eV"],
            passed=bool(
                all(
                    np.isfinite(values[k]) and values[k] <= v
                    for k, v in thresholds.items()
                )
            ),
        )


def convergence(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    checks = []
    for name in dict.fromkeys(c["geometry_name"] for c in m["calibration"]):
        rows = [c for c in m["calibration"] if c["geometry_name"] == name]
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
    report = dict(
        schema="mapho.vasp_silicon_kconvergence.v1",
        checks=checks,
        thresholds=m["thresholds"],
        passed=passed,
        selected_kppvol=m["production_kppvol"] if passed else None,
        gauge="Native V and D; no fitted energy shifts",
        scope="Sampling convergence at fixed cutoff on two calibration geometries; frontier checks use common k points.",
    )
    save_json(root / "convergence.json", report)
    return report


def execute(root):
    """Use one independently bound GPU per child, within one Slurm allocation."""
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not devices or any(not d.strip() for d in devices):
        raise RuntimeError("Run inside a GPU allocation with CUDA_VISIBLE_DEVICES set")
    deadline = time.monotonic() + 6900
    for phase in ("calibration", "samples"):
        if phase == "samples" and not convergence(root)["passed"]:
            raise RuntimeError(
                "K-point convergence failed; production inputs retained but not run"
            )
        queue = list(range(len(m[phase])))
        running = {}
        failed = []
        while queue or running:
            for device in devices:
                if device in running or not queue:
                    continue
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Allocation budget exhausted before all references completed"
                    )
                index = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
                logfile = (root / "logs" / f"{phase}_{index:03d}.log").open("w")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "mapho.commands.vasp.silicon_dataset",
                        "worker",
                        str(root),
                        phase,
                        str(index),
                    ],
                    env=env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                )
                running[device] = (process, logfile, index)
                print(f"Started {phase} {index} on allocated GPU {device}", flush=True)
            for device, (process, logfile, index) in list(running.items()):
                code = process.poll()
                if code is not None:
                    logfile.close()
                    del running[device]
                    print(f"Finished {phase} {index}: exit {code}", flush=True)
                    if code:
                        failed.append(index)
            if running:
                time.sleep(2)
        if failed:
            raise RuntimeError(f"{phase} failed for indices {failed}; see worker logs")
    reports = [
        json.loads((Path(s["directory"]) / "report.json").read_text())
        for s in m["samples"]
    ]
    completion = dict(
        schema="mapho.vasp_silicon_completion.v1",
        complete=True,
        count=len(reports),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        job=os.environ.get("SLURM_JOB_ID"),
        production_kppvol=m["production_kppvol"],
        splits={k: len(v) for k, v in m["splits"].items()},
        reports=reports,
        trained_model=False,
    )
    save_json(root / "completion.json", completion)
    return completion
