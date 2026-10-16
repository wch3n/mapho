"""Experimental serial QE 7.2 hybrid exchange exporter, separate from PBE fields.

The private GNU Fortran ABI is version gated and checked against native ACE.
Only scalar NC, eight bands, fixed occupations and full uniform meshes are exposed.
No PyTorch imports belong in this worker.
"""

from ..schema import canonical_schema

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write

from .upf import read_upf_metadata


def _json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _native_library():
    # QEpy replaces imported wrapper modules with dispatch proxies at runtime;
    # their __file__ attribute is therefore not a reliable shared-library path.
    path = importlib.metadata.distribution("qepy").locate_file(
        "qepy/qepylibs/libqepy_pw.so"
    )
    return ct.CDLL(str(path))


class NativeExchange:
    """Private QEpy 7.2 ABI. Complex k-point path; not Gamma-only, PAW or MPI."""

    def __init__(self, driver):
        if importlib.metadata.version("qepy") != "7.2.0" or driver.nproc != 1:
            raise RuntimeError("Native exchange bridge requires serial QEpy 7.2.0")
        if driver.qepy_modules.control_flags.get_gamma_only():
            raise ValueError("Use explicit complex k points, not Gamma-only storage")
        self.driver = driver
        self.lib = _native_library()
        for name in (
            "__exx_MOD_vexx",
            "__exx_MOD_vexxace_k",
            "__exx_MOD_exxinit",
            "__exx_MOD_aceinit",
        ):
            getattr(self.lib, name).restype = None
        self.lib.__getattr__("__exx_MOD_vexx").argtypes = [ct.c_void_p] * 6
        self.lib.__getattr__("__exx_MOD_vexxace_k").argtypes = [ct.c_void_p] * 5
        self.lib.__getattr__("__exx_MOD_exxinit").argtypes = [ct.c_void_p] * 2
        self.lib.__getattr__("__exx_MOD_aceinit").argtypes = [ct.c_void_p] * 2

    def rebuild(self):
        self.lib.__getattr__("__exx_MOD_exxinit")(ct.byref(ct.c_int(0)), None)
        start = time.perf_counter()
        self.lib.__getattr__("__exx_MOD_aceinit")(ct.byref(ct.c_int(0)), None)
        return time.perf_counter() - start

    def apply(self, psi, k, *, ace=False):
        psi = np.array(psi, dtype=np.complex128, order="F", copy=True)
        pw = self.driver.qepy_pw
        lda, m = psi.shape
        n = int(pw.klist.get_array_ngk()[k])
        if (
            lda != pw.wvfct.get_npwx()
            or not 1 <= m <= pw.wvfct.get_nbnd()
            or not np.isfinite(psi).all()
        ):
            raise ValueError("Invalid native plane-wave input")
        pw.wvfct.set_current_k(k + 1)
        out = np.zeros_like(psi, order="F")
        args = [ct.byref(ct.c_int(x)) for x in (lda, n, m)]
        if ace:
            energy = ct.c_double()
            self.lib.__getattr__("__exx_MOD_vexxace_k")(
                args[1],
                args[2],
                ct.c_void_p(psi.ctypes.data),
                ct.byref(energy),
                ct.c_void_p(out.ctypes.data),
            )
        else:
            self.lib.__getattr__("__exx_MOD_vexx")(
                *args, ct.c_void_p(psi.ctypes.data), ct.c_void_p(out.ctypes.data), None
            )
        return out


def reference_spec(request):
    """Validate species/hash/neutral closed-shell count before starting QE.

    Legacy single-Si requests remain readable. New requests carry one UPF per
    species and explicit cutoffs, while this pilot retains eight bands.
    """
    paths = request.get("pseudo_paths", {"Si": request.get("pseudo")})
    hashes = request.get(
        "pseudo_sha256_by_element", {"Si": request.get("pseudo_sha256")}
    )
    species = {chemical_symbols[int(z)] for z in request["numbers"]}
    if set(paths) != species or set(hashes) != species:
        raise ValueError("Pseudopotential species do not match the structure")
    metadata = {s: read_upf_metadata(path) for s, path in paths.items()}
    if any(m["element"] != s or m["sha256"] != hashes[s] for s, m in metadata.items()):
        raise ValueError("Pilot pseudopotential mismatch")
    nelect = sum(
        metadata[chemical_symbols[int(z)]]["z_valence"] for z in request["numbers"]
    )
    nocc = int(round(nelect / 2))
    if not np.isclose(nelect, 2 * nocc) or not 1 <= nocc <= 6:
        raise ValueError("Eight-band pilot requires 1--6 doubly occupied bands")
    ecut = float(request.get("ecutwfc_ry", 40))
    rho = float(request.get("ecutrho_ry", 4 * ecut))
    if not np.isfinite([ecut, rho]).all() or ecut <= 0 or rho < 4 * ecut:
        raise ValueError("Invalid norm-conserving plane-wave cutoffs")
    return paths, metadata, nocc, ecut, rho


def worker(request_path):
    from qepy.driver import Driver
    from qepy import qepy_modules

    r = json.loads(Path(request_path).read_text())
    out = Path(request_path).resolve().parent
    if r["functional"] not in ("PBE", "HSE", "PBE0"):
        raise ValueError("Unsupported exchange reference functional")
    hybrid = r["functional"] in ("HSE", "PBE0")
    mesh = tuple(r.get("kmesh", [2, 2, 2]))
    if len(mesh) != 3 or any(n not in (2, 3, 4) for n in mesh):
        raise ValueError("Use a full uniform mesh with 2--4 points per axis")
    mu = 0.0 if r["functional"] == "PBE0" else 0.106
    divergence = "gygi-baldereschi" if r["functional"] == "PBE0" else "none"
    paths, metadata, nocc, ecut, rho = reference_spec(r)
    pseudo_dir = out / "pseudos"
    pseudo_dir.mkdir(exist_ok=True)
    for symbol, path in paths.items():
        shutil.copyfile(path, pseudo_dir / f"{symbol}.upf")
    atoms = Atoms(r["numbers"], positions=r["positions"], cell=r["cell"], pbc=True)
    system = dict(
        ecutwfc=ecut,
        ecutrho=rho,
        nbnd=8,
        input_dft=r["functional"],
        occupations="fixed",
        nosym=True,
        noinv=True,
        nspin=1,
    )
    if hybrid:
        system.update(
            screening_parameter=mu,
            exx_fraction=0.25,
            nqx1=mesh[0],
            nqx2=mesh[1],
            nqx3=mesh[2],
            exxdiv_treatment=divergence,
            x_gamma_extrapolation=False,
            ace=True,
        )
    params = {
        "control": dict(
            calculation="scf",
            prefix="exchange",
            outdir=str(out / "scratch"),
            pseudo_dir=str(pseudo_dir),
            disk_io="low",
        ),
        "system": system,
        "electrons": dict(
            conv_thr=1e-10,
            electron_maxstep=150,
            mixing_beta=0.5,
            diago_full_acc=True,
            exx_maxstep=80,
        ),
    }
    write(
        out / "qe.in",
        atoms,
        format="espresso-in",
        input_data=params,
        pseudopotentials={s: f"{s}.upf" for s in paths},
        kpts=mesh,
    )
    started = time.perf_counter()
    d = Driver(str(out / "qe.in"), comm=None, logfile=str(out / "qe.log"))
    try:
        if importlib.metadata.version("f90wrap") != "0.2.14":
            raise RuntimeError("Validated f90wrap version is 0.2.14")
        d.scf()
        scf_seconds = time.perf_counter() - started
        if not d.check_convergence():
            raise RuntimeError("SCF failed; labels not written")
        # Driver convergence is the charge loop; require the outer EXX loop too.
        dexx = (
            ct.c_double.in_dll(_native_library(), "__exx_MOD_dexx").value
            if hybrid
            else 0.0
        )
        if not np.isfinite(dexx) or abs(dexx) > 1e-10:
            raise RuntimeError("Hybrid outer loop did not converge")
        p = d.qepy_pw
        bohr = float(qepy_modules.constants.get_bohr_radius_angs())
        ry = float(qepy_modules.constants.get_rytoev())
        nk, nb, lda = (
            d.get_number_of_k_points(),
            d.get_number_of_bands(),
            p.wvfct.get_npwx(),
        )
        if (nk, nb) != (np.prod(mesh), 8):
            raise ValueError("Pilot requires the full requested k mesh and eight bands")
        ngk = p.klist.get_array_ngk()[:nk].copy()
        mill = np.zeros((nk, lda, 3), dtype=np.int64)
        orbitals = np.zeros((nk, nb, lda), complex)
        eigen = np.stack([d.get_eigenvalues(k).copy() * ry for k in range(nk)])
        weights = d.get_k_point_weights().copy()
        weights /= weights.sum()
        raw_occ = np.stack([d.get_occupation_numbers(k).copy() for k in range(nk)])
        if not np.allclose(
            raw_occ[:, :nocc] / weights[:, None], 2, atol=1e-8
        ) or np.any(raw_occ[:, nocc:] > 1e-8):
            raise ValueError("Unexpected occupations")
        gap = float(eigen[:, nocc:].min() - eigen[:, :nocc].max())
        if gap <= 0:
            raise ValueError("Fixed-occupation reference is not insulating")
        report = {
            "scf_seconds": scf_seconds,
            "functional": r["functional"],
            "dexx_Ry": dexx,
            "qe_version": importlib.metadata.version("qepy"),
            "bohr_angstrom": bohr,
            "rydberg_eV": ry,
            "exchange_fraction": 0.25,
            "screening_bohr_inverse": mu,
            "exxdiv_treatment": divergence,
            "x_gamma_extrapolation": False,
            "kmesh": list(mesh),
            "pseudopotentials": metadata,
            "nocc_per_spin": nocc,
            "ecutwfc_ry": ecut,
            "ecutrho_ry": rho,
            "sampled_gap_eV": gap,
            "grid_inner_product": "discrete orthonormal FFT; sum conjugate(u)*v",
            "action_units": "eV, includes the 0.25 hybrid exchange fraction",
            "native_exact_seconds": [],
            "native_ace_seconds": [],
        }
        arrays = {}
        if hybrid:
            native = NativeExchange(d)
            report["ace_build_seconds"] = native.rebuild()
            actual_mu = ct.c_double.in_dll(
                native.lib, "__exx_base_MOD_erfc_scrlen"
            ).value
            actual_fraction = ct.c_double.in_dll(native.lib, "__exx_MOD_exxalfa").value
            if abs(actual_mu - mu) > 1e-12 or abs(actual_fraction - 0.25) > 1e-12:
                raise RuntimeError("Native exchange settings differ from the request")
            if r["functional"] == "PBE0":
                report["coulomb_zero_mode_bohr2"] = (
                    -ct.c_double.in_dll(native.lib, "__exx_base_MOD_exxdiv").value / 2
                )
            actions = np.zeros_like(orbitals)
            ace_actions = np.zeros_like(orbitals)
            probes = np.zeros((nk, 2, lda), complex)
            probe_actions = np.zeros_like(probes)
            ace_probe_actions = np.zeros_like(probes)
        for k in range(nk):
            n = int(ngk[k])
            c = np.array(d.get_pseudo_wave_function(kpt=k), order="F", copy=True)
            c[n:] = 0
            if np.max(np.abs(c.conj().T @ c - np.eye(nb))) > 1e-8:
                raise RuntimeError("Wavefunctions not orthonormal")
            orbitals[k] = c.T
            mill[k, :n] = qepy_modules.gvect.get_array_mill()[
                :, p.klist.get_array_igk_k()[:n, k] - 1
            ].T
            if hybrid:
                times = []
                for _ in range(3):
                    t = time.perf_counter()
                    a = native.apply(c, k)
                    times.append(time.perf_counter() - t)
                report["native_exact_seconds"].append(float(np.median(times)))
                times = []
                for _ in range(10):
                    t = time.perf_counter()
                    ac = native.apply(c, k, ace=True)
                    times.append(time.perf_counter() - t)
                report["native_ace_seconds"].append(float(np.median(times)))
                actions[k], ace_actions[k] = a.T * ry, ac.T * ry
                rng = np.random.default_rng(991 + k)
                z = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
                z -= c[:n] @ (c[:n].conj().T @ z)
                z = np.linalg.qr(z)[0]
                zpad = np.zeros((lda, 2), complex, order="F")
                zpad[:n] = z
                probes[k] = zpad.T
                probe_actions[k] = native.apply(zpad, k).T * ry
                ace_probe_actions[k] = native.apply(zpad, k, ace=True).T * ry
        if hybrid:
            report["ace_band_max_error_eV"] = float(
                np.max(np.abs(actions - ace_actions))
            )
            if report["ace_band_max_error_eV"] > 1e-8:
                raise RuntimeError(
                    "Native ACE does not reproduce its construction subspace"
                )
            arrays.update(
                actions=actions,
                ace_actions=ace_actions,
                probes=probes,
                probe_actions=probe_actions,
                ace_probe_actions=ace_probe_actions,
            )
        shape = tuple(map(int, d.get_number_of_grid_points()))
        arrays.update(
            orbitals=orbitals,
            mill=mill,
            ngk=ngk,
            shape=shape,
            nocc_per_spin=nocc,
            kpoints=d.get_ibz_k_points().copy(),
            weights=weights,
            eigenvalues=eigen,
            cell_bohr=d.get_ions_lattice().copy(),
            positions=atoms.positions,
            numbers=atoms.numbers,
            density=d.get_density()[:, 0].reshape(shape, order="F") / bohr**3,
            hxc=(d.get_hartree_potential() + d.get_exchange_correlation_potential())[
                :, 0
            ].reshape(shape, order="F")
            * ry,
        )
        np.savez_compressed(out / "exchange.npz", **arrays)
        report["wall_seconds"] = time.perf_counter() - started
        _json(out / "report.json", report)
    finally:
        d.stop()


def dataset(source, output, pseudo, *, resume=False):
    source, output, pseudo = (
        Path(source).resolve(),
        Path(output).resolve(),
        Path(pseudo).resolve(),
    )
    output.mkdir(parents=True, exist_ok=resume)
    original = json.loads((source / "manifest.json").read_text())
    splits = {
        key: [str(Path(p).parent) for p in original["splits"][key][:n]]
        for key, n in [("train", 8), ("validation", 2), ("test", 6)]
    }
    manifest = {
        "schema": "mapho.exchange_pilot.v1",
        "source": str(source),
        "splits": splits,
        "training_k_indices": [0, 1, 2, 3, 4, 5],
        "unseen_k_indices": [6, 7],
        "training_band_indices": [0, 1, 2, 3, 4, 5],
        "unseen_band_indices": [6, 7],
        "pseudo": str(pseudo),
        "pseudo_sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
        "scope": "HSE06 exchange correction using converged PBE occupied orbitals; 2x2x2 integration mesh; not a converged bulk gap benchmark",
    }
    if (output / "manifest.json").exists():
        if not resume or json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Cannot resume a different dataset protocol")
    else:
        _json(output / "manifest.json", manifest)
    for split, samples in splits.items():
        for name in samples:
            geometry = json.loads((source / name / "request.json").read_text())
            for functional in ("PBE", "HSE"):
                out = output / name / functional
                out.mkdir(parents=True, exist_ok=resume)
                request = {k: geometry[k] for k in ("numbers", "positions", "cell")}
                request.update(
                    functional=functional,
                    pseudo=str(pseudo),
                    pseudo_sha256=manifest["pseudo_sha256"],
                )
                if (out / "request.json").exists():
                    if json.loads((out / "request.json").read_text()) != request:
                        raise ValueError("Resume request mismatch")
                    if (out / "report.json").exists() and (
                        out / "exchange.npz"
                    ).exists():
                        continue
                else:
                    _json(out / "request.json", request)
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ):
                    env[key] = "1"
                with (out / "worker.log").open("w" if resume else "x") as f:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "mapho.qe.exchange",
                            "worker",
                            str(out / "request.json"),
                        ],
                        check=True,
                        cwd=out,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                print(f"{split} {name} {functional} complete", flush=True)
    if not (output / "complete.json").exists():
        _json(output / "complete.json", {"complete": True})


def lif_dataset(
    output,
    li_pseudo,
    f_pseudo,
    *,
    ecutwfc=100,
    lattice_constant=4.03,
    seed=29,
    resume=False,
    workers=2,
):
    """Rock-salt LiF pilot with the same perturbations/split sizes as silicon.

    Generate all 32 geometries before labels; use the first eight train, two
    validation and six test entries of the deterministic 20/6/6 split.
    """
    from ase.build import bulk

    if (
        workers not in (1, 2, 3, 4)
        or not np.isfinite(lattice_constant)
        or lattice_constant <= 0
    ):
        raise ValueError("Use 1--4 serial workers and a positive lattice constant")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    paths = {"Li": str(Path(li_pseudo).resolve()), "F": str(Path(f_pseudo).resolve())}
    hashes = {
        s: hashlib.sha256(Path(p).read_bytes()).hexdigest() for s, p in paths.items()
    }
    rng = np.random.default_rng(seed)
    order = rng.permutation(32)
    splits = {
        key: [f"sample_{i:03d}" for i in indices]
        for key, indices in [
            ("train", order[:8]),
            ("validation", order[20:22]),
            ("test", order[26:]),
        ]
    }
    geometries = {}
    for i in range(32):
        atoms = bulk("LiF", "rocksalt", a=lattice_constant)
        strain = rng.uniform(-0.006, 0.006, (3, 3))
        strain = (strain + strain.T) / 2 + np.eye(3) * rng.uniform(-0.015, 0.015)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        atoms.positions[1] += rng.uniform(-0.025, 0.025, 3)
        geometries[f"sample_{i:03d}"] = dict(
            numbers=atoms.numbers.tolist(),
            positions=atoms.positions.tolist(),
            cell=atoms.cell.array.tolist(),
        )
    settings = dict(
        pseudo_paths=paths,
        pseudo_sha256_by_element=hashes,
        ecutwfc_ry=float(ecutwfc),
        ecutrho_ry=float(4 * ecutwfc),
    )
    _, metadata, nocc, _, _ = reference_spec({**geometries["sample_000"], **settings})
    manifest = {
        "schema": "mapho.exchange_pilot.v2",
        "material": "LiF",
        "atomic_numbers": [3, 9],
        "seed": seed,
        "splits": splits,
        "geometries": geometries,
        "pseudopotentials": metadata,
        **settings,
        "nocc_per_spin": nocc,
        "lattice_constant_angstrom": lattice_constant,
        "export_workers": workers,
        "distribution": "rock-salt LiF: isotropic strain +/-1.5%, symmetric strain perturbation +/-0.6%, F displacement +/-0.025 angstrom per axis",
        "training_k_indices": list(range(6)),
        "unseen_k_indices": [6, 7],
        "training_band_indices": list(range(6)),
        "unseen_band_indices": [6, 7],
        "scope": "HSE06 exchange correction using PBE occupied orbitals; full 2x2x2 integration mesh; not a converged bulk gap benchmark",
    }
    if (output / "manifest.json").exists():
        if not resume or json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Cannot resume a different dataset protocol")
    else:
        _json(output / "manifest.json", manifest)
    jobs = []
    for split, samples in splits.items():
        for name in samples:
            for functional in ("PBE", "HSE"):
                out = output / name / functional
                out.mkdir(parents=True, exist_ok=resume)
                request = {**geometries[name], **settings, "functional": functional}
                if (out / "request.json").exists():
                    if json.loads((out / "request.json").read_text()) != request:
                        raise ValueError("Resume request mismatch")
                    if (out / "report.json").exists() and (
                        out / "exchange.npz"
                    ).exists():
                        continue
                else:
                    _json(out / "request.json", request)
                jobs.append((split, name, functional, out))

    def run(job):
        split, name, functional, out = job
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = "1"
        with (out / "worker.log").open("w" if resume else "x") as f:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mapho.qe.exchange",
                    "worker",
                    str(out / "request.json"),
                ],
                check=True,
                cwd=out,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        print(f"{split} {name} {functional} complete", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))
    if not (output / "complete.json").exists():
        _json(output / "complete.json", {"complete": True})


def unscreened_dataset(source, output, *, workers=2, resume=False):
    """Reuse matching PBE inputs and export new, full-Coulomb PBE0 targets."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if workers not in (1, 2, 3, 4) or not (source / "complete.json").exists():
        raise ValueError("Use 1--4 workers and a complete source dataset")
    original = json.loads((source / "manifest.json").read_text())
    if canonical_schema(original.get("schema")) != "mapho.exchange_pilot.v2":
        raise ValueError(
            "Expected a multi-species exchange dataset with explicit cutoffs"
        )
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {
        **original,
        "schema": "mapho.exchange_pilot.v3",
        "target_functional": "PBE0",
        "source": str(source),
        "source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        "export_workers": workers,
        "kmesh": [2, 2, 2],
        "screening_bohr_inverse": 0.0,
        "exchange_fraction": 0.25,
        "exxdiv_treatment": "gygi-baldereschi",
        "x_gamma_extrapolation": False,
        "scope": "Unscreened PBE0 exchange correction using reused PBE occupied orbitals; 2x2x2 mesh with Gaussian auxiliary-function singularity correction; no real-space Coulomb cutoff",
    }
    if (output / "manifest.json").exists():
        if not resume or json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Cannot resume a different dataset protocol")
    else:
        _json(output / "manifest.json", manifest)
    jobs = []
    for split, names in manifest["splits"].items():
        for name in names:
            old = source / name / "PBE"
            request = json.loads((old / "request.json").read_text())
            report = json.loads((old / "report.json").read_text())
            if (
                report["functional"] != "PBE"
                or report["ecutwfc_ry"] != manifest["ecutwfc_ry"]
                or report["ecutrho_ry"] != manifest["ecutrho_ry"]
                or report["pseudopotentials"] != manifest["pseudopotentials"]
            ):
                raise ValueError("Source PBE inputs do not match the intended protocol")
            reference_spec(request)
            for key in ("numbers", "positions", "cell"):
                if request[key] != manifest["geometries"][name][key]:
                    raise ValueError("Source geometry differs from the manifest")
            pbe = output / name / "PBE"
            pbe.mkdir(parents=True, exist_ok=resume)
            provenance = {"source": str(old), "sha256": {}}
            for filename in ("request.json", "report.json", "exchange.npz"):
                digest = hashlib.sha256((old / filename).read_bytes()).hexdigest()
                provenance["sha256"][filename] = digest
                if (pbe / filename).exists():
                    if (
                        hashlib.sha256((pbe / filename).read_bytes()).hexdigest()
                        != digest
                    ):
                        raise ValueError("Reused PBE file mismatch")
                else:
                    shutil.copy2(old / filename, pbe / filename)
            if not (pbe / "reuse.json").exists():
                _json(pbe / "reuse.json", provenance)
            out = output / name / "PBE0"
            out.mkdir(parents=True, exist_ok=resume)
            request = {**request, "functional": "PBE0", "kmesh": [2, 2, 2]}
            if (out / "request.json").exists():
                if json.loads((out / "request.json").read_text()) != request:
                    raise ValueError("Resume request mismatch")
                if (out / "report.json").exists() and (out / "exchange.npz").exists():
                    continue
            else:
                _json(out / "request.json", request)
            jobs.append((split, name, out))

    def run(job):
        split, name, out = job
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = "1"
        with (out / "worker.log").open("w" if resume else "x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mapho.qe.exchange",
                    "worker",
                    str(out / "request.json"),
                ],
                cwd=out,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        print(f"{split} {name} PBE0 complete", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))
    if not (output / "complete.json").exists():
        _json(output / "complete.json", {"complete": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("request", type=Path)
    d = sub.add_parser("dataset")
    d.add_argument("--source", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--pseudo", type=Path, required=True)
    d.add_argument("--resume", action="store_true")
    l = sub.add_parser("lif-dataset")
    l.add_argument("--output", type=Path, required=True)
    l.add_argument("--li-pseudo", type=Path, required=True)
    l.add_argument("--f-pseudo", type=Path, required=True)
    l.add_argument("--ecutwfc", type=float, default=100)
    l.add_argument("--lattice-constant", type=float, default=4.03)
    l.add_argument("--seed", type=int, default=29)
    l.add_argument("--workers", type=int, default=2)
    l.add_argument("--resume", action="store_true")
    u = sub.add_parser("pbe0-dataset")
    u.add_argument("--source", type=Path, required=True)
    u.add_argument("--output", type=Path, required=True)
    u.add_argument("--workers", type=int, default=2)
    u.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "worker":
        worker(a.request)
    elif a.command == "dataset":
        dataset(a.source, a.output, a.pseudo, resume=a.resume)
    elif a.command == "pbe0-dataset":
        unscreened_dataset(a.source, a.output, workers=a.workers, resume=a.resume)
    else:
        lif_dataset(
            a.output,
            a.li_pseudo,
            a.f_pseudo,
            ecutwfc=a.ecutwfc,
            lattice_constant=a.lattice_constant,
            seed=a.seed,
            resume=a.resume,
            workers=a.workers,
        )


if __name__ == "__main__":
    main()
