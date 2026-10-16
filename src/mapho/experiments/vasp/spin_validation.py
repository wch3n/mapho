"""Native collinear PBE PAW validation, including spin exchange and scalar reuse."""

import copy
import json
from pathlib import Path
import shutil

from ase.build import bulk
import numpy as np

from ...vasp.paw import (
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


def prepare(
    root, potential_root, installed_executable, bridge_executable, scalar_operator
):
    root = Path(root).resolve()
    if (root / "study.json").exists():
        raise FileExistsError("Study already prepared")
    atoms = bulk("Fe", "bcc", a=2.86)
    potcars = {"Fe": Path(potential_root) / "Fe/POTCAR"}
    options = dict(encut=600.0, mesh=(2, 2, 2), nbands=12, algorithm="Exact", ispin=2)
    reference = root / "Fe/reference"
    m = prepare_reference(reference, atoms, potcars, magmoms=[2.5], **options)
    control = root / "Fe/unpatched"
    control.mkdir()
    for f in ("POSCAR", "POTCAR", "INCAR", "KPOINTS"):
        shutil.copyfile(reference / f, control / f)
    cm = copy.deepcopy(m)
    cm["unpatched_control"] = True
    save_json(control / "manifest.json", cm)
    reverse = root / "Fe/reversed_native"
    rm = prepare_reference(reverse, atoms, potcars, magmoms=[-2.5], **options)
    rm["unpatched_control"] = True
    (reverse / "MLHAM_EXPORT").unlink()
    save_json(reverse / "manifest.json", rm)
    save_json(
        root / "study.json",
        dict(
            schema="mapho.vasp_paw_spin_validation.v1",
            installed_executable=str(Path(installed_executable).resolve()),
            bridge_executable=str(Path(bridge_executable).resolve()),
            scalar_operator=str(Path(scalar_operator).resolve()),
            tolerance_eV=1e-4,
            purpose="collinear interface validation; not magnetic/cutoff/k-mesh convergence",
        ),
    )


def validate(root):
    root = Path(root).resolve()
    m = json.loads((root / "study.json").read_text())
    report = dict(schema=m["schema"], checks={}, complete=False, passed=False)
    save_json(root / "validation.json", report)
    launcher = [
        "mpirun",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,vader",
        "-np",
        "1",
    ]

    def execute(folder, executable):
        print("Running", folder, flush=True)
        return run(folder, executable, launcher=launcher)

    def record(key, value):
        report["checks"][key] = value
        save_json(root / "validation.json", report)
        print(key, json.dumps(value), flush=True)

    ref = root / "Fe/reference"
    native = root / "Fe/unpatched"
    execute(native, m["installed_executable"])
    execute(ref, m["bridge_executable"])
    record("unpatched_vs_bridge", compare(native, ref, tolerance_eV=m["tolerance_eV"]))
    bands = read_bands(ref)
    moment = bands["spin_moment_muB"]
    record(
        "magnetic_reference",
        dict(
            moment_muB=moment,
            spin_electrons=bands["spin_electrons"].tolist(),
            max_spin_splitting_eV=float(
                np.abs(np.diff(bands["eigenvalues_eV"], axis=0)).max()
            ),
            passed=bool(
                abs(moment) > 0.5 and abs(bands["spin_electrons"].sum() - 8.0) < 1e-5
            ),
        ),
    )
    operator = root / "Fe/operator.h5"
    export_operator(ref, operator)
    for tag, kwargs in (
        ("replay", {}),
        ("common_shift", dict(shift_eV=0.25)),
        ("spin_dependent_shift", dict(shift_eV=[0.17, -0.09])),
        ("reversed_operator", dict(reverse_spins=True)),
    ):
        directory = root / "Fe" / tag
        prepare_replay(operator, directory, **kwargs)
        execute(directory, m["bridge_executable"])
        record(tag, compare(ref, directory, tolerance_eV=m["tolerance_eV"], **kwargs))
    reversed_native = root / "Fe/reversed_native"
    execute(reversed_native, m["installed_executable"])
    record(
        "native_spin_reversal",
        compare(
            ref, reversed_native, reverse_spins=True, tolerance_eV=m["tolerance_eV"]
        ),
    )
    reversed_moment = read_bands(root / "Fe/reversed_operator")["spin_moment_muB"]
    record(
        "moment_reversal",
        dict(
            reference_muB=moment,
            replay_muB=reversed_moment,
            passed=bool(abs(moment + reversed_moment) < 1e-5),
        ),
    )

    kpoints = np.array([[0.13, 0.21, 0.07], [0.31, 0.11, 0.23], [0.43, 0.27, 0.17]])
    replay_k = root / "Fe/new_k_replay"
    native_k = root / "Fe/new_k_native"
    prepare_replay(operator, replay_k, kpoints=kpoints)
    native_k.mkdir()
    for f in ("POSCAR", "POTCAR", "KPOINTS"):
        shutil.copyfile(replay_k / f, native_k / f)
    shutil.copyfile(ref / "CHGCAR", native_k / "CHGCAR")
    settings = json.loads((replay_k / "manifest.json").read_text())["settings"]
    settings["ICHARG"] = 11
    write_incar(native_k / "INCAR", settings)
    save_json(
        native_k / "manifest.json",
        dict(
            unpatched_control=True,
            settings=settings,
            inputs={
                f: sha(native_k / f)
                for f in ("POSCAR", "POTCAR", "KPOINTS", "INCAR", "CHGCAR")
            },
        ),
    )
    execute(native_k, m["installed_executable"])
    execute(replay_k, m["bridge_executable"])
    record("new_k_replay", compare(native_k, replay_k, tolerance_eV=m["tolerance_eV"]))

    # A V1 HDF5 file and V1 matrix format must still work with the new bridge.
    scalar = root / "Si_legacy_replay"
    replay_manifest = prepare_replay(m["scalar_operator"], scalar)
    execute(scalar, m["bridge_executable"])
    import h5py

    with h5py.File(m["scalar_operator"]) as f:
        old = json.loads(f.attrs["metadata_json"])
    record(
        "ispin1_backwards_compatibility",
        compare(old["reference"], scalar, tolerance_eV=m["tolerance_eV"]),
    )
    report["complete"] = True
    report["passed"] = all(check["passed"] for check in report["checks"].values())
    save_json(root / "validation.json", report)
    if not report["passed"]:
        raise RuntimeError("Collinear acceptance failed; see validation.json")
    return report
