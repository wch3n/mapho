"""Serial QEpy backend: export fields, solve a frozen potential, and validate."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

from ..fields import (
    FIELD_UNITS,
    FieldSample,
    AtomicReference,
    load_operator_reference,
    from_qe_grid,
    to_qe_grid,
)
from .upf import read_upf_metadata


SI_URL = (
    "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.2.upf"
)
SI_SHA256 = "73d21061e1050ef8134ba50a20e61b9d0bcef98d0a9e0a110bbd076e47282eee"


@dataclass
class QEConfig:
    ecutwfc_ry: float = 40.0
    ecutrho_ry: float = 160.0
    kmesh: tuple = (2, 2, 2)
    nbnd: int = 8
    conv_thr_ry: float = 1e-12
    diagonal_thr_ry: float = 1e-12
    electron_maxstep: int = 150

    def validate(self):
        if (
            not np.isfinite(
                [
                    self.ecutwfc_ry,
                    self.ecutrho_ry,
                    self.conv_thr_ry,
                    self.diagonal_thr_ry,
                ]
            ).all()
            or self.ecutwfc_ry <= 0
            or self.ecutrho_ry < 4 * self.ecutwfc_ry
            or self.conv_thr_ry <= 0
            or self.diagonal_thr_ry <= 0
        ):
            raise ValueError("Invalid cutoffs or convergence thresholds")
        if (
            len(self.kmesh) != 3
            or any(int(k) != k or k < 1 for k in self.kmesh)
            or int(self.nbnd) != self.nbnd
            or self.nbnd < 1
            or int(self.electron_maxstep) != self.electron_maxstep
            or self.electron_maxstep < 1
        ):
            raise ValueError("Invalid k mesh, number of bands, or iteration count")


def _json(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fetch_silicon_pseudo(path):
    """Download the unchanged SG15 file, requiring the recorded content hash."""
    path = Path(path)
    data = urllib.request.urlopen(SI_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SI_SHA256:
        raise ValueError("SG15 download checksum mismatch")
    with path.open("xb") as stream:
        stream.write(data)
    return read_upf_metadata(path)


def prepare_reference(
    atoms, pseudopotentials, output, config=None, *, mode="reference"
):
    """Prepare an isolated calculation; no electronic engine is imported here."""
    config = config or QEConfig()
    config.validate()
    if mode not in ("reference", "atomic"):
        raise ValueError("Unsupported preparation mode")
    if (
        not np.all(atoms.pbc)
        or np.linalg.det(atoms.cell.array) <= 1e-10
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("Expected a finite, right-handed, fully periodic structure")
    symbols = atoms.get_chemical_symbols()
    if set(symbols) != set(pseudopotentials):
        raise ValueError("Supply exactly one pseudopotential for every element")
    pseudos = {s: read_upf_metadata(p) for s, p in pseudopotentials.items()}
    if any(pseudos[s]["element"] != s for s in pseudos):
        raise ValueError("Pseudopotential element mismatch")
    nelect = sum(pseudos[s]["z_valence"] for s in symbols)
    if not np.isclose(nelect % 2, 0) or config.nbnd < nelect / 2:
        raise ValueError(
            "Fixed occupations require an even electron count and sufficient bands"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "pseudos").mkdir()
    paths = {}
    for symbol, path in pseudopotentials.items():
        target = output / "pseudos" / f"{symbol}.upf"
        shutil.copyfile(path, target)
        paths[symbol] = str(target)
    request = {
        "mode": mode,
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pseudopotentials": pseudos,
        "pseudo_paths": paths,
        "config": asdict(config),
        "nelect": nelect,
    }
    _json(output / "request.json", request)
    return output / "request.json"


def _launch(request):
    """A process per solve avoids sharing QE's global Fortran state."""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    output = Path(request).resolve().parent
    with (output / "worker.log").open("x") as log:
        try:
            subprocess.run(
                [sys.executable, "-m", "mapho.qe", "_worker", str(request)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
                cwd=output,
                timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"QE worker failed; inspect {output / 'worker.log'} "
                f"and {output / 'qe.log'}"
            ) from exc


def run_reference(atoms, pseudopotentials, output, config=None):
    request = prepare_reference(atoms, pseudopotentials, output, config)
    _launch(request)
    return FieldSample.load(request.parent / "fields.npz")


def run_atomic_reference(atoms, pseudopotentials, output, config=None):
    """Initialize QE's atomic superposition and export inputs without an SCF step."""
    request = prepare_reference(atoms, pseudopotentials, output, config, mode="atomic")
    _launch(request)
    return AtomicReference.load(request.parent / "atomic.npz")


def solve(
    reference_path, output, hxc=None, constant_eV=0.0, cosine_eV=0.0, *, kpoints=None
):
    """Solve a full Hxc field; explicit fractional k points request bands only.

    The default reconstructs density on the reference integration mesh.
    Explicit points return eigenvalues without integrating a path density.
    """
    sample = load_operator_reference(reference_path)
    output = Path(output).resolve()
    if isinstance(sample, AtomicReference) and hxc is None:
        raise ValueError(
            "An atomic reference requires an explicit predicted full Hxc field"
        )
    potential = (
        sample.hxc.copy() if hxc is None else np.asarray(hxc, dtype=np.float64).copy()
    )
    if potential.shape != sample.shape or not np.isfinite(potential).all():
        raise ValueError("Hxc potential must be finite and match the reference grid")
    if not np.isfinite([constant_eV, cosine_eV]).all():
        raise ValueError("Non-finite perturbation")
    if kpoints is not None:
        kpoints = np.asarray(kpoints, dtype=np.float64)
        if (
            kpoints.ndim != 2
            or kpoints.shape[1] != 3
            or len(kpoints) == 0
            or not np.isfinite(kpoints).all()
        ):
            raise ValueError(
                "Band k points must be a finite nonempty (N, 3) fractional array"
            )
    potential += constant_eV
    x = np.arange(sample.shape[0]) / sample.shape[0]
    potential += cosine_eV * np.cos(2 * np.pi * x)[:, None, None]
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "input_hxc_eV.npy", potential, allow_pickle=False)
    request = {
        "mode": "solve" if kpoints is None else "bands",
        "numbers": sample.numbers.tolist(),
        "positions": sample.positions.tolist(),
        "cell": sample.cell.tolist(),
        "pseudopotentials": sample.metadata["pseudopotentials"],
        "pseudo_paths": {
            s: str((Path(reference_path).resolve().parent / p).resolve())
            for s, p in sample.metadata["pseudo_paths"].items()
        },
        "config": sample.metadata["qe_config"],
        "nelect": sample.metadata["nelect"],
        "reference_path": str(Path(reference_path).resolve()),
    }
    if kpoints is not None:
        request["kpoints"] = kpoints.tolist()
    _json(output / "request.json", request)
    _launch(output / "request.json")
    result = "solution.npz" if kpoints is None else "bands.npz"
    with np.load(output / result, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _eigenvalues(driver):
    return np.stack(
        [
            driver.get_eigenvalues(k).copy()
            for k in range(driver.get_number_of_k_points())
        ]
    )


def _spectrum(driver):
    nk = driver.get_number_of_k_points()
    eigen = _eigenvalues(driver)
    weights = driver.get_k_point_weights().copy()
    weights /= weights.sum()
    weighted_occ = np.stack(
        [driver.get_occupation_numbers(k).copy() for k in range(nk)]
    )
    return eigen, weights, weighted_occ / weights[:, None]


def _worker(request_path):
    # These imports MUST stay inside the separate worker. No PyTorch dependency.
    from qepy.driver import Driver
    from qepy import qepy_modules

    versions = {
        n: importlib.metadata.version(n) for n in ("qepy", "f90wrap", "numpy", "ase")
    }
    if versions["qepy"] != "7.2.0" or versions["f90wrap"] != "0.2.14":
        raise RuntimeError("Validated backend requires qepy==7.2.0 and f90wrap==0.2.14")
    request_path = Path(request_path).resolve()
    out = request_path.parent
    request = json.loads(request_path.read_text())
    config = QEConfig(**request["config"])
    config.validate()
    for symbol, path in request["pseudo_paths"].items():
        info = read_upf_metadata(path)
        if info["sha256"] != request["pseudopotentials"][symbol]["sha256"]:
            raise ValueError(f"Pseudopotential changed: {symbol}")
    atoms = Atoms(
        request["numbers"],
        positions=request["positions"],
        cell=request["cell"],
        pbc=True,
    )
    # All files live in this run. No density/wavefunction restart is used.
    pseudo_dir = out / "worker_pseudos"
    pseudo_dir.mkdir()
    for symbol, path in request["pseudo_paths"].items():
        shutil.copyfile(path, pseudo_dir / f"{symbol}.upf")
    params = {
        "control": {
            "calculation": "scf",
            "prefix": "mapho",
            "restart_mode": "from_scratch",
            "outdir": str(out / "scratch"),
            "pseudo_dir": str(pseudo_dir),
            "disk_io": "low",
        },
        "system": {
            "ecutwfc": config.ecutwfc_ry,
            "ecutrho": config.ecutrho_ry,
            "nbnd": config.nbnd,
            "input_dft": "PBE",
            "occupations": "fixed",
            "nosym": True,
            "noinv": True,
            "nspin": 1,
        },
        "electrons": {
            "conv_thr": config.conv_thr_ry,
            "electron_maxstep": config.electron_maxstep,
            "mixing_beta": 0.5,
            "startingpot": "atomic",
            "startingwfc": "random",
            "diago_full_acc": True,
        },
    }
    kgrid = config.kmesh
    if request["mode"] == "bands":
        points = np.asarray(request["kpoints"], dtype=float)
        # Positive placeholder weights initialize QE; a path is never used to
        # integrate density or occupations. Eigenvalue stability is checked below.
        kgrid = np.column_stack([points, np.full(len(points), 1 / len(points))])
    write(
        out / "qe.in",
        atoms,
        format="espresso-in",
        input_data=params,
        pseudopotentials={s: f"{s}.upf" for s in request["pseudo_paths"]},
        kpts=kgrid,
    )
    started = time.perf_counter()
    driver = Driver(str(out / "qe.in"), comm=None, logfile=str(out / "qe.log"))
    try:
        if driver.nproc != 1:
            raise RuntimeError("Only serial QEpy is validated")
        bohr = qepy_modules.constants.get_bohr_radius_angs()
        ry = qepy_modules.constants.get_rytoev()
        shape = tuple(map(int, driver.get_number_of_grid_points()))
        grid = lambda a: from_qe_grid(a, shape)
        initial_density = grid(driver.get_density()) / bohr**3
        initial_hxc = (
            grid(
                driver.get_hartree_potential()
                + driver.get_exchange_correlation_potential()
            )
            * ry
        )
        ionic = grid(driver.get_local_pp()) * ry
        if request["mode"] == "atomic":
            metadata = {
                "schema": "mapho.atomic_reference.v1",
                "nspin": 1,
                "functional": "PBE",
                "field_units": {
                    n: FIELD_UNITS[n] for n in ("reference_density", "reference_hxc")
                },
                "length_unit": "angstrom",
                "energy_unit": "eV",
                "pbc": [True, True, True],
                "grid_convention": "fractional_ijk_endpoint_excluded",
                "grid_origin_fractional": [0, 0, 0],
                "potential_gauge": "native QE periodic convention; preserve absolute Hxc offset",
                "reference_definition": "Hxc and density of QE's normalized initial atomic superposition",
                "pseudopotentials": request["pseudopotentials"],
                "pseudo_paths": {
                    s: f"pseudos/{s}.upf" for s in request["pseudo_paths"]
                },
                "qe_config": asdict(config),
                "nelect": request["nelect"],
                "runtime": versions,
            }
            AtomicReference(
                atoms.numbers,
                driver.get_ions_positions().copy() * bohr,
                driver.get_ions_lattice().copy() * bohr,
                {"reference_density": initial_density, "reference_hxc": initial_hxc},
                driver.get_ibz_k_points().copy(),
                metadata,
            ).save(out / "atomic.npz")
        elif request["mode"] == "reference":
            driver.scf()
            if not driver.check_convergence():
                raise RuntimeError(
                    "Reference SCF did not converge; no training label written"
                )
            fields = {
                "density": grid(driver.get_density()) / bohr**3,
                "reference_density": initial_density,
                "reference_hxc": initial_hxc,
                "ionic": ionic,
                "hartree": grid(driver.get_hartree_potential()) * ry,
                "xc": grid(driver.get_exchange_correlation_potential()) * ry,
            }
            eigen, weights, occupations = _spectrum(driver)
            metadata = {
                "schema": "mapho.scalar_fields.v1",
                "field_units": FIELD_UNITS,
                "length_unit": "angstrom",
                "energy_unit": "eV",
                "grid_convention": "fractional_ijk_endpoint_excluded",
                "grid_origin_fractional": [0, 0, 0],
                "pbc": [True, True, True],
                "nspin": 1,
                "functional": "PBE",
                "nelect": request["nelect"],
                "potential_gauge": "native QE periodic convention; preserve absolute Hxc offset",
                "reference_definition": "Hxc and density of QE's normalized initial atomic superposition",
                "pseudopotentials": request["pseudopotentials"],
                "pseudo_paths": {
                    s: f"pseudos/{s}.upf" for s in request["pseudo_paths"]
                },
                "qe_config": asdict(config),
                "runtime": versions,
                "qe_bohr_angstrom": bohr,
                "qe_rydberg_eV": ry,
                "scf_steps": driver.get_scf_steps(),
                "scf_error_ry": float(driver.get_scf_error()),
            }
            sample = FieldSample(
                atoms.numbers,
                driver.get_ions_positions().copy() * bohr,
                driver.get_ions_lattice().copy() * bohr,
                fields,
                driver.get_ibz_k_points().copy(),
                weights,
                eigen * ry,
                occupations,
                metadata,
            )
            sample.save(out / "fields.npz")
            centered, offset = sample.residual()
            np.savez_compressed(
                out / "target.npz", residual_eV=centered, mean_offset_eV=offset
            )
        else:
            sample = load_operator_reference(request["reference_path"])
            if shape != sample.shape:
                raise ValueError("QE grid differs from reference")
            expected_kpoints = np.asarray(request.get("kpoints", sample.kpoints))
            actual_kpoints = driver.get_ibz_k_points().copy()
            if (
                not np.allclose(
                    driver.get_ions_lattice() * bohr, sample.cell, atol=1e-10, rtol=0
                )
                or actual_kpoints.shape != expected_kpoints.shape
                or not np.allclose(actual_kpoints, expected_kpoints, atol=1e-10, rtol=0)
            ):
                raise ValueError("QE cell or k points differ from requested values")
            potential = np.load(out / "input_hxc_eV.npy", allow_pickle=False)
            driver.set_external_potential(
                to_qe_grid(potential / ry), exttype=("hartree", "xc")
            )
            # Explicitly rebuild the operator before diagonalization. The default
            # external-potential mode ADDS a field and would double count Hxc.
            used = driver.create_array()
            driver.qepy_pw.qepy_mod.qepy_calc_effective_potential(used, gather=True)
            applied = grid(used) * ry
            if not np.allclose(applied, ionic + potential, atol=1e-10, rtol=0):
                raise RuntimeError(
                    "Applied potential differs from ionic + supplied Hxc"
                )
            previous = None
            for iteration in range(1, 7):
                driver.qepy_pw.qepy_mod.qepy_diagonalize(
                    iter=iteration, threshold=config.diagonal_thr_ry
                )
                eigen = _eigenvalues(driver) * ry
                if not np.isfinite(eigen).all():
                    raise RuntimeError("Non-finite eigenvalues")
                if previous is not None and np.max(np.abs(eigen - previous)) < 1e-8:
                    break
                previous = eigen.copy()
            else:
                raise RuntimeError("Frozen-potential eigenvalues failed to stabilize")
            if request["mode"] == "bands":
                np.savez_compressed(
                    out / "bands.npz",
                    eigenvalues_eV=eigen,
                    kpoints_fractional=actual_kpoints,
                    cell_angstrom=sample.cell,
                    nelect=request["nelect"],
                    potential_gauge=sample.metadata["potential_gauge"],
                    applied_local_eV=applied,
                    input_hxc_eV=potential,
                    diagonalization_passes=iteration,
                    eigenvalue_last_change_eV=np.max(np.abs(eigen - previous)),
                )
            else:
                # Construct density/occupations directly; never perform an SCF update.
                driver.qepy_pw.qepy_mod.qepy_calc_density()
                eigen, weights, occupations = _spectrum(driver)
                density = grid(driver.get_density()) / bohr**3
                if (
                    not np.isfinite(density).all()
                    or abs(density.sum() * sample.dv - request["nelect"]) > 1e-5
                ):
                    raise RuntimeError(
                        "Frozen-potential density has incorrect electron count"
                    )
                np.savez_compressed(
                    out / "solution.npz",
                    eigenvalues_eV=eigen * ry,
                    density=density,
                    initial_density=initial_density,
                    kweights=weights,
                    occupations=occupations,
                    applied_local_eV=applied,
                    input_hxc_eV=potential,
                    diagonalization_passes=iteration,
                )
        _json(
            out / "runtime.json",
            {
                "versions": versions,
                "python": sys.version,
                "wall_seconds": time.perf_counter() - started,
                "engine": "serial QEpy/Quantum ESPRESSO 7.2",
            },
        )
    finally:
        driver.stop()


def validate_reference(reference_path, output):
    sample = FieldSample.load(reference_path)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    replay = solve(reference_path, out / "replay")
    shift = 0.2
    shifted = solve(reference_path, out / "constant", constant_eV=shift)
    perturbed = solve(reference_path, out / "cosine", cosine_eV=0.5)
    eigen_error = float(np.max(np.abs(replay["eigenvalues_eV"] - sample.eigenvalues)))
    density_error = float(
        np.sum(np.abs(replay["density"] - sample.fields["density"]))
        * sample.dv
        / sample.metadata["nelect"]
    )
    shift_error = float(
        np.max(np.abs(shifted["eigenvalues_eV"] - replay["eigenvalues_eV"] - shift))
    )
    shift_density = float(
        np.sum(np.abs(shifted["density"] - replay["density"]))
        * sample.dv
        / sample.metadata["nelect"]
    )
    perturb_density = float(
        np.sum(np.abs(perturbed["density"] - replay["density"]))
        * sample.dv
        / sample.metadata["nelect"]
    )
    report = {
        "reference": str(Path(reference_path).resolve()),
        "scope": "backend numerical validation, not ML accuracy or DFT cutoff convergence",
        "eigenvalue_max_error_eV": eigen_error,
        "density_relative_integrated_absolute_error": density_error,
        "constant_shift_eV": shift,
        "constant_shift_max_error_eV": shift_error,
        "constant_shift_density_relative_error": shift_density,
        "cosine_amplitude_eV": 0.5,
        "cosine_density_relative_change": perturb_density,
        "initial_density_relative_error": float(
            np.sum(np.abs(replay["initial_density"] - sample.fields["density"]))
            * sample.dv
            / sample.metadata["nelect"]
        ),
        "tolerances": {
            "eigenvalues_eV": 1e-4,
            "density_relative": 1e-5,
            "constant_shift_eV": 1e-5,
            "cosine_min_density_change": 1e-4,
        },
        "passed": (
            eigen_error < 1e-4
            and density_error < 1e-5
            and shift_error < 1e-5
            and shift_density < 1e-5
            and perturb_density > 1e-4
        ),
    }
    _json(out / "validation.json", report)
    if not report["passed"]:
        raise RuntimeError(
            f"Numerical validation failed; see {out / 'validation.json'}"
        )
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "_worker":
        if len(argv) != 2:
            raise ValueError("Expected exactly one worker request path")
        _worker(argv[1])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser(
        "silicon", help="Run Si reference and frozen-potential validation"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument(
        "--pseudo", type=Path, help="NC/PBE Si UPF v2; default downloads pinned SG15"
    )
    ref = sub.add_parser(
        "reference", help="Export scalar fields from a neutral PBE/NC calculation"
    )
    atomic = sub.add_parser(
        "atomic", help="Initialize inference inputs without running SCF"
    )
    for p in (ref, atomic):
        p.add_argument("--structure", type=Path, required=True)
        p.add_argument(
            "--pseudo", action="append", required=True, help="Element=/path/to/file.upf"
        )
        p.add_argument("--output", type=Path, required=True)
    for p in (demo, ref, atomic):
        p.add_argument("--ecutwfc", type=float, default=40)
        p.add_argument("--kmesh", type=int, nargs=3, default=(2, 2, 2))
        p.add_argument("--nbnd", type=int, default=8)
    check = sub.add_parser("validate")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    frozen = sub.add_parser(
        "solve", help="Solve a full Hxc field, preserving QE's ionic operators"
    )
    frozen.add_argument("--reference", type=Path, required=True)
    frozen.add_argument(
        "--hxc", type=Path, help="NPY scalar full Hxc in eV on the reference grid"
    )
    frozen.add_argument("--output", type=Path, required=True)
    bands = sub.add_parser(
        "bands", help="Diagonalize a frozen potential at explicit k points"
    )
    bands.add_argument("--reference", type=Path, required=True)
    bands.add_argument(
        "--hxc", type=Path, help="Predicted full Hxc NPY; default is the SCF reference"
    )
    bands.add_argument(
        "--kpoints", type=Path, required=True, help="NPY (N, 3) fractional k points"
    )
    bands.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("solve", "bands"):
        hxc = None if args.hxc is None else np.load(args.hxc, allow_pickle=False)
        kpoints = (
            np.load(args.kpoints, allow_pickle=False)
            if args.command == "bands"
            else None
        )
        solve(args.reference, args.output, hxc, kpoints=kpoints)
        print(
            args.output / ("bands.npz" if args.command == "bands" else "solution.npz")
        )
    elif args.command == "validate":
        print(json.dumps(validate_reference(args.reference, args.output), indent=2))
    else:
        config = QEConfig(
            ecutwfc_ry=args.ecutwfc,
            ecutrho_ry=4 * args.ecutwfc,
            kmesh=tuple(args.kmesh),
            nbnd=args.nbnd,
        )
        config.validate()
        if args.command == "silicon":
            args.output.mkdir(parents=True, exist_ok=False)
            pseudo = args.pseudo
            if pseudo is None:
                pseudo = args.output / "Si_ONCV_PBE-1.2.upf"
                info = fetch_silicon_pseudo(pseudo)
                _json(args.output / "pseudo_source.json", {"url": SI_URL, **info})
            run_reference(
                bulk("Si", "diamond", a=5.43),
                {"Si": pseudo},
                args.output / "reference",
                config,
            )
            report = validate_reference(
                args.output / "reference" / "fields.npz", args.output / "validation"
            )
            print(json.dumps(report, indent=2))
        else:
            pseudos = {}
            for value in args.pseudo:
                symbol, path = value.split("=", 1)
                if symbol in pseudos:
                    raise ValueError(f"Duplicate pseudopotential for {symbol}")
                pseudos[symbol] = path
            if args.command == "atomic":
                run_atomic_reference(read(args.structure), pseudos, args.output, config)
                print(args.output / "atomic.npz")
            else:
                run_reference(read(args.structure), pseudos, args.output, config)
                print(args.output / "fields.npz")


if __name__ == "__main__":
    main()
