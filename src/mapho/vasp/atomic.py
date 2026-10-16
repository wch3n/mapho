"""Prepare and validate atomic PAW references without electronic diagonalization."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .paw import prepare_reference, read_paw, read_pot, save_json, sha, write_incar


MARKER = "MAPHO_ATOMIC_EXPORT"
COMPLETION = "MAPHO_ATOMIC_DONE.json"
MESSAGE = "MAPHO: atomic reference exported without diagonalization"
COMPLETION_RECORD = dict(
    schema="mapho.vasp_atomic_export.v1",
    complete=True,
    diagonalized=False,
    electronic_steps=0,
    icharg=12,
    ispin=1,
)
INPUTS = ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
OUTPUTS = ("POT", "MAPHO_PAW.dat", COMPLETION, "vasprun.xml", "OUTCAR")


def enable_export_only(folder, manifest):
    """Opt a prepared atomic reference into the private initialization-only hook."""
    folder = Path(folder)
    settings = manifest["settings"]
    settings.update(LWAVE=False, LCHARG=False)
    write_incar(folder / "INCAR", settings)
    (folder / MARKER).touch()
    manifest.update(
        atomic_export_only=True,
        purpose="Fixed atomic-density PAW operator export without diagonalization",
        baseline_definition="ICHARG=12 atomic density and occupations; no electronic optimization",
        inputs={n: sha(folder / n) for n in (*INPUTS, "MAPHO_EXPORT", MARKER)},
    )
    validate_request(folder, manifest)
    save_json(folder / "manifest.json", manifest)
    return manifest


def prepare_atomic_reference(
    folder, atoms, potcars, *, encut=600.0, nbands=8, grid_shape=None, export_only=True
):
    """Build a geometry-only scalar PBE baseline, with an optional legacy solve."""
    if grid_shape is not None and (
        len(grid_shape) != 3 or any(int(n) != n or n < 1 for n in grid_shape)
    ):
        raise ValueError("Atomic reference requires three positive FFT dimensions")
    manifest = prepare_reference(
        folder, atoms, potcars, mesh=(1, 1, 1), encut=encut, nbands=nbands
    )
    settings = manifest["settings"]
    settings.update(ICHARG=12, NELMDL=0, NELMIN=16, LWAVE=False, LCHARG=False)
    if grid_shape is not None:
        settings.update(dict(zip(("NGXF", "NGYF", "NGZF"), map(int, grid_shape))))
    write_incar(Path(folder) / "INCAR", settings)
    manifest.update(
        purpose="Geometry-only fixed atomic-density PAW baseline",
        baseline_definition="ICHARG=12, ISTART=0, no input CHGCAR/WAVECAR",
        inputs={n: sha(Path(folder) / n) for n in INPUTS},
    )
    if export_only:
        return enable_export_only(folder, manifest)
    save_json(Path(folder) / "manifest.json", manifest)
    return manifest


def validate_request(folder, manifest):
    """Reject settings or electronic inputs inconsistent with an atomic baseline."""
    folder = Path(folder)
    settings = manifest["settings"]
    required = dict(ICHARG=12, ISTART=0, NSW=0, ISPIN=1, GGA="PE", KPAR=1, NCORE=1)
    if manifest.get("atomic_export_only") is not True or any(
        settings.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Export-only mode requires a scalar PBE atomic reference")
    forbidden = (
        "LHFCALC",
        "LDAU",
        "LNONCOLLINEAR",
        "LSORBIT",
        "LREAL",
        "LH5",
        "LWAVE",
        "LCHARG",
        "LEPSILON",
        "LCALCPOL",
        "LPEAD",
    )
    if any(settings.get(key, False) is not False for key in forbidden) or settings.get(
        "METAGGA", "None"
    ) not in (None, "None", "NONE"):
        raise ValueError("Unsupported settings for export-only atomic reference")
    for name in ("MAPHO_EXPORT", MARKER):
        if not (folder / name).is_file() or manifest["inputs"].get(name) != sha(
            folder / name
        ):
            raise ValueError(f"Missing or changed atomic export marker: {name}")
    for name in (
        "CHGCAR",
        "WAVECAR",
        "vaspwave.h5",
        "MAPHO_IMPORT",
        "MAPHO_ACE_IMPORT",
        "MAPHO_FOCK_EXPORT",
        "MAPHO_ACE_EXPORT",
        "POT",
        "MAPHO_PAW.dat",
        "OUTCAR",
        "vasprun.xml",
        "EIGENVAL",
        COMPLETION,
    ):
        if (folder / name).exists():
            raise ValueError(f"Atomic export must start without {name}")


def validate_result(folder):
    """Require positive completion evidence and absence of electronic iterations."""
    folder = Path(folder)
    report = json.loads((folder / COMPLETION).read_text())
    if any(
        type(report.get(k)) is not type(v) or report[k] != v
        for k, v in COMPLETION_RECORD.items()
    ):
        raise ValueError("Invalid completion record for atomic export")
    if MESSAGE not in (folder / "stdout.log").read_text():
        raise ValueError("Atomic export hook was not executed")
    tree = ET.parse(folder / "vasprun.xml")
    if tree.findall(".//scstep") or tree.findall(".//eigenvalues"):
        raise ValueError(
            "Export-only reference unexpectedly performed electronic optimization"
        )
    read_pot(folder / "POT")
    read_paw(folder / "MAPHO_PAW.dat")
    return {name: sha(folder / name) for name in OUTPUTS}
