"""Scalar and collinear spin-polarized PBE PAW export/replay in VASP.

The native solver retains the kinetic energy, PAW projectors and overlap.
This interface supplies the total smooth potential and full on-site D matrices.
It is an operator/data interface, not a trained PAW Hamiltonian model.
"""

from ..schema import canonical_schema

from io import StringIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from ase.io import read, write
from ase.io.vasp import read_vasp_configuration

from ..file_utils import sha


SCHEMA = "mapho.vasp_paw_operator.v1"
SPIN_SCHEMA = "mapho.vasp_paw_operator.v2"
COMPONENT_SCHEMA = "mapho.vasp_paw_hybrid_components.v1"


def hybrid_reference(manifest):
    """Recognize hybrid inputs without treating the string '.FALSE.' as true."""
    flag = manifest.get("settings", {}).get("LHFCALC", False)
    enabled = str(flag).strip().upper() in ("TRUE", ".TRUE.", "T", "1")
    return enabled or manifest.get("functional", "").upper() in (
        "PBE0",
        "HSE",
        "HSE03",
        "HSE06",
    )


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_pot(path):
    """Read native POT in eV; scalar xyz or collinear (up,down,xyz).

    POT stores two spin potentials, unlike CHGCAR's charge/magnetization
    convention. Parse both augmentation blocks so the second field cannot
    accidentally be treated as an opaque nonmagnetic suffix.
    """
    text = Path(path).read_text()
    stream = StringIO(text)
    atoms = read_vasp_configuration(stream)
    header = text[: stream.tell()]

    def nonblank():
        line = stream.readline()
        while line and not line.strip():
            line = stream.readline()
        if not line:
            raise ValueError("Truncated POT")
        return line

    def grid():
        shape = tuple(map(int, nonblank().split()))
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError("Invalid POT grid")
        values = []
        while len(values) < np.prod(shape):
            values.extend(float(x.replace("D", "E")) for x in nonblank().split())
        if len(values) != np.prod(shape) or not np.isfinite(values).all():
            raise ValueError("Invalid POT grid values")
        return np.array(values).reshape(shape, order="F"), stream.tell()

    def augmentation():
        for atom in range(len(atoms)):
            match = re.fullmatch(
                r"augmentation occupancies\s+(\d+)\s+(\d+)\s*", nonblank().strip()
            )
            if not match or int(match[1]) != atom + 1 or int(match[2]) < 1:
                raise ValueError("Invalid/missing POT PAW augmentation record")
            count = int(match[2])
            values = []
            while len(values) < count:
                values.extend(float(x.replace("D", "E")) for x in nonblank().split())
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError("Invalid POT augmentation values")

    up, suffix_start = grid()
    augmentation()
    if not text[stream.tell() :].strip():
        return atoms, up, header, text[suffix_start:]
    moments = []
    while len(moments) < len(atoms):
        moments.extend(float(x.replace("D", "E")) for x in nonblank().split())
    if len(moments) != len(atoms) or not np.isfinite(moments).all():
        raise ValueError("Invalid POT spin separator")
    second_start = stream.tell()
    down, second_suffix_start = grid()
    augmentation()
    if up.shape != down.shape or text[stream.tell() :].strip():
        raise ValueError(
            "Only one scalar or two matching collinear POT grids are supported"
        )
    return (
        atoms,
        np.stack([up, down]),
        header,
        [text[suffix_start:second_start], text[second_suffix_start:]],
    )


def write_pot(path, field, header, suffix):
    field = np.asarray(field)
    if field.ndim == 3:
        grids, suffixes = field[None], [suffix]
    elif (
        field.ndim == 4
        and field.shape[0] == 2
        and isinstance(suffix, (list, tuple))
        and len(suffix) == 2
    ):
        grids, suffixes = field, suffix
    else:
        raise ValueError(
            "Potential must be xyz or (2,nx,ny,nz) with two native suffixes"
        )
    if (
        not all(isinstance(s, str) for s in suffixes)
        or not np.isfinite(field).all()
        or np.iscomplexobj(field)
    ):
        raise ValueError(
            "Potential must be finite and real with native augmentation text"
        )
    with Path(path).open("w") as out:
        out.write(header.rstrip() + "\n\n")
        for grid, tail in zip(grids, suffixes):
            out.write(" ".join(map(str, grid.shape)) + "\n")
            v = grid.ravel(order="F")
            for start in range(0, v.size, 5):
                # Native INCHG reads fixed-width (1X,E17.11) records.
                entries = [f"{x:17.10E}" for x in v[start : start + 5]]
                if any(len(s) != 17 for s in entries):
                    raise ValueError("Potential exceeds native POT exponent range")
                out.write("".join(" " + s for s in entries) + "\n")
            out.write(tail)


def read_paw(path, *, d_symmetry_tolerance_eV=1e-10):
    if not np.isfinite(d_symmetry_tolerance_eV) or d_symmetry_tolerance_eV < 0:
        raise ValueError("Invalid on-site symmetry tolerance")
    with Path(path).open() as f:
        version = f.readline().strip()
        dims = list(map(int, f.readline().split()))
        if version == "MLHAM_PAW_V1" and len(dims) == 3:
            lmdim, natoms, ntypes = dims
            nspin = 1
        elif version == "MLHAM_PAW_V2" and len(dims) == 4 and dims[3] == 2:
            lmdim, natoms, ntypes, nspin = dims
        else:
            raise ValueError("Unsupported PAW matrix format/spin dimensions")
        if min(lmdim, natoms, ntypes) < 1:
            raise ValueError("Invalid PAW dimensions")
        channels = []
        for _ in range(ntypes):
            n = int(f.readline())
            ls = list(map(int, f.readline().split()))
            if (
                not ls
                or len(ls) != n
                or min(ls) < 0
                or sum(2 * l + 1 for l in ls) > lmdim
            ):
                raise ValueError("Invalid PAW projector channels")
            channels.append(ls)
        d = np.zeros((nspin, natoms, lmdim, lmdim))
        q = np.zeros((natoms, lmdim, lmdim))
        types = []
        for a in range(natoms):
            ia, it, n = map(int, f.readline().split())
            if (
                ia != a + 1
                or not 1 <= it <= ntypes
                or n != sum(2 * l + 1 for l in channels[it - 1])
            ):
                raise ValueError("PAW atom/channel mismatch")
            types.append(it - 1)
            for j in range(n):
                for i in range(n):
                    values = list(map(float, f.readline().split()))
                    if len(values) != nspin + 1:
                        raise ValueError("Incomplete PAW spin matrix record")
                    d[:, a, i, j], q[a, i, j] = values[:nspin], values[-1]
        if f.read().strip():
            raise ValueError("Unexpected trailing PAW data")
    for matrix, tolerance in ((d, d_symmetry_tolerance_eV), (q, 1e-10)):
        if not np.isfinite(matrix).all() or not np.allclose(
            matrix, matrix.swapaxes(-1, -2), atol=tolerance, rtol=0
        ):
            raise ValueError("PAW matrices must be finite and symmetric")
    return dict(
        d=d[0] if nspin == 1 else d, q=q, types=np.array(types), channels=channels
    )


def write_paw(path, paw):
    d, q = np.asarray(paw["d"]), np.asarray(paw["q"])
    dd = d[None] if d.ndim == 3 else d
    if (
        dd.ndim != 4
        or dd.shape[0] not in (1, 2)
        or q.ndim != 3
        or dd.shape[1:] != q.shape
        or q.shape[-1] != q.shape[-2]
    ):
        raise ValueError("Invalid PAW spin/matrix shape")
    for matrix in (dd, q):
        if (
            np.iscomplexobj(matrix)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-10, rtol=0)
        ):
            raise ValueError("PAW matrices must be finite, real and symmetric")
    types = np.asarray(paw["types"])
    channels = paw["channels"]
    if types.shape != (len(q),) or not np.issubdtype(types.dtype, np.integer):
        raise ValueError("Provide one integer PAW type index per atom")
    if not channels or any(
        not ls or any(int(l) != l or l < 0 for l in ls) for ls in channels
    ):
        raise ValueError("Invalid PAW channel angular momenta")
    if np.any(types < 0) or np.any(types >= len(channels)):
        raise ValueError("Invalid PAW type index")
    for a, t in enumerate(types):
        n = sum(2 * l + 1 for l in channels[t])
        if n > q.shape[-1]:
            raise ValueError("Insufficient PAW matrix size")
        if (
            np.any(dd[:, a, n:, :])
            or np.any(dd[:, a, :, n:])
            or np.any(q[a, n:, :])
            or np.any(q[a, :, n:])
        ):
            raise ValueError("Nonzero entries in padded PAW channels")
    nspin = dd.shape[0]
    with Path(path).open("w") as f:
        dims = f"{q.shape[-1]} {len(q)} {len(channels)}"
        f.write(
            ("MLHAM_PAW_V1\n" + dims + "\n")
            if nspin == 1
            else ("MLHAM_PAW_V2\n" + dims + " 2\n")
        )
        for ls in channels:
            f.write(str(len(ls)) + "\n" + " ".join(map(str, ls)) + "\n")
        for a, t in enumerate(types):
            n = sum(2 * l + 1 for l in channels[t])
            f.write(f"{a + 1} {t + 1} {n}\n")
            for j in range(n):
                for i in range(n):
                    f.write(
                        " ".join(f"{x:.17E}" for x in [*dd[:, a, i, j], q[a, i, j]])
                        + "\n"
                    )


def potcar_info(path, symbol):
    text = Path(path).read_text()
    title = re.search(r"TITEL\s*=\s*([^\n]+)", text)
    lexch = re.search(r"LEXCH\s*=\s*(\w+)", text)
    if (
        not title
        or not lexch
        or lexch[1] != "PE"
        or not title[1].startswith("PAW_PBE ")
    ):
        raise ValueError(f"{path}: requires a PBE PAW dataset")
    if title[1].split()[1].split("_")[0] != symbol or text.count("End of Dataset") != 1:
        raise ValueError(f"{path}: wrong element or multiple datasets")
    return dict(
        symbol=symbol,
        path=str(Path(path).resolve()),
        sha256=sha(path),
        title=title[1].strip(),
        zval=float(re.search(r"ZVAL\s*=\s*([\d.]+)", text)[1]),
        enmax_eV=float(re.search(r"ENMAX\s*=\s*([\d.]+)", text)[1]),
    )


def prepare_reference(
    folder,
    atoms,
    potcars,
    *,
    encut=600.0,
    mesh=(4, 4, 4),
    nbands=12,
    algorithm="Normal",
    ispin=1,
    magmoms=None,
    nupdown=None,
):
    folder = Path(folder).resolve()
    if ispin not in (1, 2) or not np.all(atoms.pbc):
        raise ValueError(
            "Only fully periodic scalar ISPIN=1 or collinear ISPIN=2 is supported"
        )
    moments = np.asarray(
        atoms.get_initial_magnetic_moments() if magmoms is None else magmoms,
        dtype=float,
    )
    if moments.shape != (len(atoms),) or not np.isfinite(moments).all():
        raise ValueError(
            "Provide one finite collinear MAGMOM per atom; spin vectors are unsupported"
        )
    if ispin == 1 and (np.any(moments) or nupdown is not None):
        raise ValueError("Magnetic moments/spin constraints require ISPIN=2")
    symbols = atoms.get_chemical_symbols()
    order = list(dict.fromkeys(symbols))
    if symbols != [s for s in order for _ in range(symbols.count(s))]:
        raise ValueError("Group atoms by element before preparing a reference")
    if set(potcars) != set(order):
        raise ValueError("Provide one PAW dataset for each element")
    infos = [potcar_info(potcars[s], s) for s in order]
    if not np.isfinite(encut) or encut < max(p["enmax_eV"] for p in infos):
        raise ValueError("ENCUT must meet every PAW dataset ENMAX")
    nelect = sum(symbols.count(p["symbol"]) * p["zval"] for p in infos)
    if nupdown is not None and (not np.isfinite(nupdown) or not 0 <= nupdown <= nelect):
        raise ValueError(
            "NUPDOWN must be between zero and the electron count, or omitted"
        )
    max_occupied = (
        nelect / 2
        if ispin == 1
        else (nelect + nupdown) / 2
        if nupdown is not None
        else nelect
    )
    if (ispin == 1 and nelect % 2) or int(nbands) != nbands or nbands <= max_occupied:
        raise ValueError(
            "Provide enough bands for occupied and empty states in every spin channel"
        )
    if len(mesh) != 3 or any(int(n) != n or n < 1 for n in mesh):
        raise ValueError("Invalid k mesh")
    if algorithm not in ("Normal", "Exact"):
        raise ValueError("Supported solvers are Normal and Exact")
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    with (folder / "POTCAR").open("wb") as f:
        for s in order:
            f.write(Path(potcars[s]).read_bytes())
    (folder / "KPOINTS").write_text(
        "mapho PBE PAW reference\n0\nGamma\n" + " ".join(map(str, mesh)) + "\n0 0 0\n"
    )
    settings = dict(
        SYSTEM="mapho PBE PAW",
        GGA="PE",
        ISPIN=ispin,
        LNONCOLLINEAR=False,
        LSORBIT=False,
        LHFCALC=False,
        LDAU=False,
        ENCUT=encut,
        PREC="Accurate",
        LREAL=False,
        LASPH=True,
        ADDGRID=False,
        ISYM=-1,
        NBANDS=nbands,
        ISMEAR=0,
        SIGMA=0.05,
        ALGO=algorithm,
        EDIFF=1e-9,
        NELM=160,
        NELMIN=6,
        NELMDL=-5,
        NSW=0,
        IBRION=-1,
        ISTART=0,
        ICHARG=2,
        LMAXMIX=4,
        KPAR=1,
        NCORE=1,
        LWAVE=True,
        LCHARG=True,
        LVTOT=False,
        LH5=False,
    )
    if ispin == 2:
        settings["MAGMOM"] = " ".join(f"{x:.16g}" for x in moments)
        if nupdown is not None:
            settings["NUPDOWN"] = float(nupdown)
    write_incar(folder / "INCAR", settings)
    (folder / "MLHAM_EXPORT").touch()
    manifest = dict(
        schema="mapho.vasp_paw_reference.v1",
        functional="PBE",
        spin="nonmagnetic" if ispin == 1 else "collinear",
        initial_moments_muB=moments.tolist(),
        nupdown=nupdown,
        settings=settings,
        kmesh=list(mesh),
        nelect=nelect,
        potentials=infos,
        inputs={n: sha(folder / n) for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")},
        purpose="operator replay validation; not cutoff or k-point convergence",
    )
    save_json(folder / "manifest.json", manifest)
    return manifest


def write_incar(path, settings):
    def value(v):
        return (".TRUE." if v else ".FALSE.") if isinstance(v, bool) else str(v)

    Path(path).write_text("".join(f"{k} = {value(v)}\n" for k, v in settings.items()))


def read_bands(folder):
    root = ET.parse(Path(folder) / "vasprun.xml").getroot()
    calculations = root.findall("calculation")
    if not calculations:
        raise ValueError("No electronic calculation found")
    # ISPIN=1 and spinors both have one eigenvalue set: reject spinors explicitly.
    for tag in ("LNONCOLLINEAR", "LSORBIT"):
        if any(
            (el.text or "").strip().upper() in ("T", ".TRUE.", "TRUE")
            for el in root.findall(f'.//i[@name="{tag}"]')
        ):
            raise ValueError("Spinor/SOC eigenvalues are unsupported")
    spins = calculations[-1].findall("./eigenvalues/array/set/set")
    nspin = len(spins)
    if nspin not in (1, 2):
        raise ValueError("Expected one scalar or two collinear eigenvalue sets")
    bands = np.array(
        [
            [
                [list(map(float, r.text.split())) for r in k.findall("r")]
                for k in spin.findall("set")
            ]
            for spin in spins
        ]
    )
    kpoints = np.array(
        [
            list(map(float, v.text.split()))
            for v in root.findall('./kpoints/varray[@name="kpointlist"]/v')
        ]
    )
    if (
        bands.ndim != 4
        or bands.shape[1] != len(kpoints)
        or bands.shape[-1] != 2
        or not np.isfinite(bands).all()
    ):
        raise ValueError("Invalid VASP bands")
    weight_nodes = root.findall('./kpoints/varray[@name="weights"]/v')
    weights = (
        np.array([float(v.text) for v in weight_nodes])
        if weight_nodes
        else np.full(len(kpoints), 1 / len(kpoints))
    )
    if (
        weights.shape != (len(kpoints),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-7)
    ):
        raise ValueError("Invalid k-point weights")
    # EIGENVAL has six decimals in this build; XML energies only four.
    with (Path(folder) / "EIGENVAL").open() as f:
        for _ in range(5):
            next(f)
        _, nk, nb = map(int, next(f).split())
        if (nk, nb) != bands.shape[1:3]:
            raise ValueError("EIGENVAL/XML dimensions differ")
        energies = np.empty((nspin, nk, nb))
        occupations = np.empty_like(energies)
        for k in range(nk):
            line = next(f)
            while not line.strip():
                line = next(f)
            coords = np.array(list(map(float, line.split())))
            if coords.shape != (4,) or not np.allclose(
                coords[:3], kpoints[k], atol=1e-7, rtol=0
            ):
                raise ValueError("EIGENVAL/XML k points differ")
            for n in range(nb):
                parts = next(f).split()
                if len(parts) != 1 + 2 * nspin or int(parts[0]) != n + 1:
                    raise ValueError("Invalid collinear EIGENVAL record")
                energies[:, k, n] = list(map(float, parts[1 : 1 + nspin]))
                occupations[:, k, n] = list(map(float, parts[1 + nspin :]))
    if not np.isfinite(energies).all() or not np.allclose(
        energies, bands[:, :, :, 0], atol=5.1e-5, rtol=0
    ):
        raise ValueError("EIGENVAL/XML energies differ beyond printed precision")
    if not np.isfinite(occupations).all() or not np.allclose(
        occupations, bands[:, :, :, 1], atol=5.1e-5, rtol=0
    ):
        raise ValueError("EIGENVAL/XML occupations differ beyond printed precision")
    result = dict(
        eigenvalues_eV=energies[0] if nspin == 1 else energies,
        occupations=occupations[0] if nspin == 1 else occupations,
        kpoints=kpoints,
        weights=weights,
        nspin=nspin,
    )
    if nspin == 2:
        electrons = np.einsum("skb,k->s", occupations, weights)
        result["spin_electrons"] = electrons
        result["spin_moment_muB"] = float(electrons[0] - electrons[1])
    return result


def run(folder, executable, launcher=(), timeout=1500):
    folder, executable = Path(folder).resolve(), Path(executable).resolve()
    if (folder / "run.json").exists():
        raise FileExistsError("Use a fresh run directory")
    m = json.loads((folder / "manifest.json").read_text())
    for name, expected in m["inputs"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Input changed since preparation: {name}")
    result = dict(
        executable=str(executable),
        executable_sha256=sha(executable),
        command=[*launcher, str(executable)],
        status="running",
    )
    save_json(folder / "run.json", result)
    start = time.monotonic()
    try:
        with (folder / "stdout.log").open("w") as output:
            cp = subprocess.run(
                result["command"],
                cwd=folder,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        result["returncode"] = cp.returncode
        log = (folder / "stdout.log").read_text()
        outcar = (folder / "OUTCAR").read_text() if (folder / "OUTCAR").exists() else ""
        if cp.returncode or "General timing and accounting" not in outcar:
            raise RuntimeError(f"VASP did not finish successfully: {folder}")
        if "aborting loop because EDIFF is reached" not in outcar:
            raise RuntimeError(f"Electronic convergence not certified: {folder}")
        marker = (
            "full PAW D imported; native Q verified"
            if m.get("replay")
            else "full PAW D and Q exported"
        )
        if not m.get("unpatched_control") and marker not in log:
            raise RuntimeError(
                "VASP bridge marker missing; check executable and sentinels"
            )
        read_bands(folder)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - start
        save_json(folder / "run.json", result)
    return result


def check_pot_geometry(atoms, poscar):
    """Check a rounded native POT header against a full-precision structure."""
    if atoms.numbers.tolist() != poscar.numbers.tolist():
        raise ValueError("POT and reference structure differ")
    delta = atoms.get_scaled_positions(wrap=False) - poscar.get_scaled_positions(
        wrap=False
    )
    delta -= np.rint(delta)
    if not np.allclose(atoms.cell, poscar.cell, atol=6e-7, rtol=0) or not np.allclose(
        delta, 0, atol=6e-7, rtol=0
    ):
        raise ValueError("POT and reference structure differ")


def export_operator(folder, output, *, components_only=False):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    m = json.loads((folder / "manifest.json").read_text())
    hybrid = hybrid_reference(m)
    if hybrid and not components_only:
        raise ValueError(
            "Hybrid POT and PAW D omit the nonlocal Fock operator; use components_only=True for diagnostics"
        )
    if components_only and not hybrid:
        raise ValueError("components_only is reserved for incomplete hybrid exports")
    status = json.loads((folder / "run.json").read_text())
    if status["status"] != "complete" or m.get("replay") or m.get("unpatched_control"):
        raise ValueError("Export requires a completed reference SCF with the bridge")
    for name, expected in m["inputs"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Reference input changed: {name}")
    atoms, field, header, suffix = read_pot(folder / "POT")
    poscar = read(folder / "POSCAR", format="vasp")
    # Native OUTPOS wraps ions and prints 6 decimals in the POT header. Check
    # periodic equivalence at that precision, then retain the hashed input's
    # full-precision geometry for the operator and all later solves.
    check_pot_geometry(atoms, poscar)
    atoms = poscar
    stream = StringIO()
    write(stream, atoms, format="vasp", direct=True, vasp5=True)
    header = stream.getvalue()
    # Keep raw D in incomplete hybrid archives. Small native numerical
    # asymmetry is recorded, not silently projected away at export time.
    paw = read_paw(
        folder / "MLHAM_PAW.dat", d_symmetry_tolerance_eV=1e-6 if hybrid else 1e-10
    )
    bands = read_bands(folder)
    if len(paw["types"]) != len(atoms):
        raise ValueError("PAW atom count differs")
    nspin = m["settings"]["ISPIN"]
    if (
        (1 if field.ndim == 3 else field.shape[0]) != nspin
        or (1 if paw["d"].ndim == 3 else paw["d"].shape[0]) != nspin
        or bands["nspin"] != nspin
    ):
        raise ValueError("Potential, D, bands, and input spin counts disagree")
    schema = COMPONENT_SCHEMA if hybrid else SCHEMA if nspin == 1 else SPIN_SCHEMA
    metadata = dict(
        schema=schema,
        reference=str(folder),
        reference_manifest=m,
        run=status,
        nspin=nspin,
        spin_channels=["scalar"] if nspin == 1 else ["up", "down"],
        source_sha256={
            n: sha(folder / n)
            for n in ("POT", "MLHAM_PAW.dat", "vasprun.xml", "EIGENVAL", "OUTCAR")
        },
        field_units="eV",
        grid_axes="xyz, fractional coordinates i/n",
        on_site_basis="native VASP projector order; radial channel then native real spherical harmonics",
        gauge="native exported total-potential gauge; D and potential must be changed consistently",
    )
    if hybrid:
        metadata.update(
            complete_hamiltonian=False,
            omitted_components=["nonlocal Fock operator"],
            raw_d_antisymmetry_max_eV=float(
                np.max(np.abs(paw["d"] - paw["d"].swapaxes(-1, -2)))
            ),
            d_symmetry_tolerance_eV=1e-6,
            purpose="Exact smooth/PAW components from a hybrid SCF; not a complete hybrid Hamiltonian",
        )
    with h5py.File(output, "x") as f:
        f.attrs["schema"] = schema
        f.attrs["metadata_json"] = json.dumps(metadata)
        f.create_dataset("potential_eV", data=field, compression="gzip")
        for key in ("d", "q", "types"):
            f.create_dataset("paw/" + key, data=paw[key])
        f["paw"].attrs["channels_json"] = json.dumps(paw["channels"])
        f.create_dataset("cell_A", data=np.asarray(atoms.cell))
        f.create_dataset("positions_A", data=atoms.positions)
        f.create_dataset("atomic_numbers", data=atoms.numbers)
        for key, value in bands.items():
            f.create_dataset("bands/" + key, data=value)
        f.create_dataset("native_pot_header", data=header)
        if nspin == 1:
            f.create_dataset("native_pot_suffix", data=suffix)
        else:
            f.create_dataset(
                "native_pot_suffixes", data=suffix, dtype=h5py.string_dtype()
            )
    return metadata


def prepare_replay(
    operator,
    folder,
    *,
    shift_eV=0.0,
    potential=None,
    d=None,
    kpoints=None,
    reverse_spins=False,
    omit_fock=False,
):
    operator, folder = Path(operator).resolve(), Path(folder).resolve()
    with h5py.File(operator, "r") as f:
        schema = canonical_schema(f.attrs["schema"])
        if schema not in (SCHEMA, SPIN_SCHEMA, COMPONENT_SCHEMA):
            raise ValueError("Wrong operator schema")
        m = json.loads(f.attrs["metadata_json"])
        hybrid = hybrid_reference(m["reference_manifest"])
        partial = schema == COMPONENT_SCHEMA
        if (hybrid or partial) and not (hybrid and partial and omit_fock):
            raise ValueError(
                "Hybrid components cannot replay a full Hamiltonian; omit_fock=True explicitly enables the diagnostic"
            )
        if omit_fock and not partial:
            raise ValueError(
                "omit_fock requires an explicitly incomplete hybrid component export"
            )
        field = f["potential_eV"][:]
        paw = {key: f["paw/" + key][:] for key in ("d", "q", "types")}
        paw["channels"] = json.loads(f["paw"].attrs["channels_json"])
        header = f["native_pot_header"].asstr()[()]
        suffix = (
            f["native_pot_suffix"].asstr()[()]
            if field.ndim == 3
            else f["native_pot_suffixes"].asstr()[:].tolist()
        )
    nspin = 1 if field.ndim == 3 else field.shape[0]
    if nspin not in (1, 2) or paw["d"].shape != (
        (nspin,) + paw["q"].shape if nspin == 2 else paw["q"].shape
    ):
        raise ValueError("Inconsistent operator spin/matrix dimensions")
    if potential is not None:
        if np.shape(potential) != field.shape:
            raise ValueError("Predicted potential grid differs from reference")
        field = np.asarray(potential).copy()
    if d is not None:
        if np.shape(d) != paw["d"].shape:
            raise ValueError("Predicted PAW D shape differs from reference")
        paw["d"] = np.asarray(d).copy()
    d_hermitization_max_eV = 0.0
    if omit_fock and d is None:
        # The diagnostic solver requires a Hermitian operator. Preserve raw D
        # in the archive and report this bounded numerical projection in the
        # replay manifest; never apply it to arbitrary learned predictions.
        anti = paw["d"] - paw["d"].swapaxes(-1, -2)
        if not np.isfinite(anti).all() or np.max(np.abs(anti)) > 1e-6:
            raise ValueError(
                "Hybrid on-site asymmetry exceeds the diagnostic tolerance"
            )
        d_hermitization_max_eV = float(np.max(np.abs(anti)) * 0.5)
        paw["d"] = 0.5 * (paw["d"] + paw["d"].swapaxes(-1, -2))
    shifts = np.asarray(shift_eV, dtype=float)
    if shifts.ndim == 0:
        shifts = np.repeat(shifts, nspin)
    if shifts.shape != (nspin,) or not np.isfinite(shifts).all():
        raise ValueError("Supply a finite common shift or one shift per spin channel")
    if reverse_spins:
        if nspin != 2:
            raise ValueError("Spin reversal requires ISPIN=2")
        if m["reference_manifest"]["settings"].get("NUPDOWN", 0) != 0:
            raise ValueError(
                "Cannot reverse a nonzero fixed NUPDOWN with this interface"
            )
        field = field[::-1].copy()
        paw["d"] = paw["d"][::-1].copy()
    # In a generalized PAW eigenproblem delta H = c S, not c I:
    # delta V=c and delta D=c Q must be applied together.
    field = field + (shifts[0] if nspin == 1 else shifts[:, None, None, None])
    paw["d"] = paw["d"] + (
        shifts[0] * paw["q"]
        if nspin == 1
        else shifts[:, None, None, None] * paw["q"][None]
    )
    ref = Path(m["reference"])
    manifest = m["reference_manifest"]
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        if sha(ref / name) != manifest["inputs"][name]:
            raise ValueError(f"Reference {name} changed")
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("POSCAR", "POTCAR", "KPOINTS"):
        shutil.copyfile(ref / name, folder / name)
    if kpoints is not None:
        k = np.asarray(kpoints)
        if k.ndim != 2 or k.shape[1] != 3 or not len(k) or not np.isfinite(k).all():
            raise ValueError("Explicit k points must have shape (n, 3)")
        (folder / "KPOINTS").write_text(
            "mapho frozen PAW operator\n"
            + str(len(k))
            + "\nReciprocal\n"
            + "".join(" ".join(f"{x:.16g}" for x in row) + " 1\n" for row in k)
        )
    settings = dict(manifest["settings"])
    if omit_fock:
        # Freeze the exported components. Do not ask VASP to construct exchange
        # from random/reoptimized orbitals and label it a replay of the reference.
        for key in (
            "AEXX",
            "HFSCREEN",
            "HFRCUT",
            "PRECFOCK",
            "LFOCKACE",
            "TIME",
            "ALDAX",
            "ALDAC",
            "AGGAX",
            "AGGAC",
        ):
            settings.pop(key, None)
        settings.update(LHFCALC=False, ALGO="Exact")
    if reverse_spins and "MAGMOM" in settings:
        settings["MAGMOM"] = " ".join(
            str(-float(x)) for x in settings["MAGMOM"].split()
        )
    settings.update(
        ICHARG=14,
        ISTART=0,
        NELMDL=0,
        LWAVE=False,
        LCHARG=False,
        NGXF=field.shape[-3],
        NGYF=field.shape[-2],
        NGZF=field.shape[-1],
    )
    if settings.get("ALGO") == "Exact":
        settings["NELMIN"] = 2
    write_incar(folder / "INCAR", settings)
    write_pot(folder / "POT", field, header, suffix)
    write_paw(folder / "MLHAM_PAW.dat", paw)
    (folder / "MLHAM_IMPORT").touch()
    m = dict(
        schema="mapho.vasp_paw_replay.v1",
        replay=True,
        operator=str(operator),
        operator_sha256=sha(operator),
        shift_eV=float(shifts[0]) if nspin == 1 else shifts.tolist(),
        reverse_spins=reverse_spins,
        settings=settings,
        inputs={
            n: sha(folder / n)
            for n in ("POSCAR", "POTCAR", "INCAR", "KPOINTS", "POT", "MLHAM_PAW.dat")
        },
    )
    if omit_fock:
        m.update(
            complete_hamiltonian=False,
            diagnostic="hybrid smooth and on-site components with Fock omitted",
            d_hermitization_max_eV=d_hermitization_max_eV,
            target_functional=manifest["functional"],
            omitted_components=["nonlocal Fock operator"],
        )
    save_json(folder / "manifest.json", m)
    return m


def compare(reference, replay, shift_eV=0.0, tolerance_eV=1e-4, reverse_spins=False):
    a, b = read_bands(reference), read_bands(replay)
    if a["eigenvalues_eV"].shape != b["eigenvalues_eV"].shape or not np.allclose(
        a["kpoints"], b["kpoints"], atol=1e-10, rtol=0
    ):
        raise ValueError("Band or k-point mismatch")
    ref = a["eigenvalues_eV"]
    if reverse_spins:
        if a["nspin"] != 2:
            raise ValueError("Spin reversal requires two spin channels")
        ref = ref[::-1]
    shift = np.asarray(shift_eV, dtype=float)
    if shift.ndim == 1 and a["nspin"] == 2 and shift.shape == (2,):
        shift = shift[:, None, None]
    elif shift.ndim != 0:
        raise ValueError("Invalid spin-dependent comparison shift")
    errors = b["eigenvalues_eV"] - ref - shift
    return dict(
        max_abs_eV=float(np.abs(errors).max()),
        mae_eV=float(np.abs(errors).mean()),
        tolerance_eV=tolerance_eV,
        passed=bool(np.abs(errors).max() < tolerance_eV),
        nkpoints=errors.shape[-2],
        nbands=errors.shape[-1],
        nspin=a["nspin"],
        per_spin_max_abs_eV=np.abs(errors).reshape(a["nspin"], -1).max(axis=1).tolist(),
        expected_shift_eV=np.asarray(shift_eV).tolist(),
        reverse_spins=reverse_spins,
    )
