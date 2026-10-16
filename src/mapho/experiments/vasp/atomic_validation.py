"""Compare initialization-only and diagonalized atomic PAW references."""

import json
from pathlib import Path

from ase.io import read
import h5py
import numpy as np

from ...vasp.atomic import prepare_atomic_reference
from ...vasp.paw import compare, export_operator, prepare_replay, run, save_json, sha
from .training_validation import LAUNCHER


def validate(root, references, executable):
    root, executable = Path(root).resolve(), Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    save_json(
        root / "manifest.json",
        dict(
            executable=str(executable),
            executable_sha256=sha(executable),
            references={k: str(Path(v).resolve()) for k, v in references.items()},
            potential_tolerance_eV=1e-8,
            d_tolerance_eV=1e-8,
            q_tolerance=1e-12,
            band_tolerance_eV=1e-4,
        ),
    )
    for label, source in references.items():
        if not label or Path(label).name != label or label in (".", ".."):
            raise ValueError("Reference labels must be simple directory names")
        source = Path(source).resolve()
        native = json.loads((source / "manifest.json").read_text())
        atoms = read(source / "POSCAR", format="vasp")
        potcars = {p["symbol"]: p["path"] for p in native["potentials"]}
        for p in native["potentials"]:
            if sha(p["path"]) != p["sha256"]:
                raise ValueError("Reference PAW dataset changed")
        with h5py.File(source / "operator.h5") as archive:
            shape = archive["potential_eV"].shape
        folder = root / label
        timings = {}
        for mode, fast in (("legacy", False), ("export_only", True)):
            path = folder / mode
            prepare_atomic_reference(
                path,
                atoms,
                potcars,
                encut=native["settings"]["ENCUT"],
                nbands=native["settings"]["NBANDS"],
                grid_shape=shape,
                export_only=fast,
            )
            timings[mode] = run(path, executable, launcher=LAUNCHER, timeout=1800)[
                "wall_seconds"
            ]
            export_operator(path, path / "operator.h5")
        row = dict(label=label, natoms=len(atoms), wall_seconds=timings)
        with (
            h5py.File(folder / "legacy/operator.h5") as old,
            h5py.File(folder / "export_only/operator.h5") as new,
        ):
            for key in ("cell_A", "positions_A", "atomic_numbers", "paw/types"):
                np.testing.assert_array_equal(old[key][:], new[key][:])
            assert "bands" not in new
            for key, metric in (
                ("potential_eV", "potential_max_abs_eV"),
                ("paw/d", "d_max_abs_eV"),
                ("paw/q", "q_max_abs"),
            ):
                row[metric] = float(np.max(np.abs(old[key][:] - new[key][:])))
        row["operator_passed"] = (
            row["potential_max_abs_eV"] <= 1e-8
            and row["d_max_abs_eV"] <= 1e-8
            and row["q_max_abs"] <= 1e-12
        )
        save_json(folder / "comparison.json", row)
        if not row["operator_passed"]:
            raise RuntimeError(f"Atomic operator changed for {label}: {row}")
        prepare_replay(folder / "export_only/operator.h5", folder / "replay")
        run(folder / "replay", executable, launcher=LAUNCHER, timeout=1800)
        row["bands"] = compare(folder / "legacy", folder / "replay", tolerance_eV=1e-4)
        row["passed"] = row["operator_passed"] and row["bands"]["passed"]
        row["speedup"] = timings["legacy"] / timings["export_only"]
        save_json(folder / "comparison.json", row)
        rows.append(row)
        save_json(root / "results.json", rows)
        if not row["passed"]:
            raise RuntimeError(f"Atomic replay differs for {label}")
    result = dict(
        complete=bool(rows),
        passed=bool(rows) and all(r["passed"] for r in rows),
        rows=rows,
    )
    save_json(root / "completion.json", result)
    return result
