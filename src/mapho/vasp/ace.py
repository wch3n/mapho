"""PAW plus externally constructed ACE; scalar, one rank, fixed reciprocal mesh.

ACE is exact on its construction subspace, not on arbitrary orbitals. Its
factors include the nonlocal PAW exchange action and exchange fraction;
the matching exported D retains VASP's on-site contribution.
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np

from .paw import (
    export_operator,
    prepare_replay,
    read_bands,
    save_json,
    sha,
    write_incar,
)


SCHEMA = "mapho.vasp_paw_ace.v2"
MAGIC = b"MAPHO_ACE_V1".ljust(16)
SAMPLES_MAGIC = b"MAPHO_FOCK_V1".ljust(16)


def _read_transport(path, magic, block_names):
    """Read MAPHO's little-endian coefficient/action transport format."""
    data = memoryview(Path(path).read_bytes())
    offset = 0

    def take(dtype, count):
        nonlocal offset
        size = np.dtype(dtype).itemsize * count
        if count < 0 or offset + size > len(data):
            raise ValueError("Truncated ACE archive")
        result = np.frombuffer(data[offset : offset + size], dtype=dtype).copy()
        offset += size
        return result

    if bytes(take("u1", 16)) != magic:
        raise ValueError("Unsupported ACE archive header")
    dims = take("<i4", 4)
    nr, nb, nk, ns = map(int, dims)
    grid = take("<i4", 3)
    cell = take("<f8", 9).reshape(3, 3, order="F").T
    fraction = float(take("<f8", 1)[0])
    if (
        min(dims) < 1
        or ns != 1
        or np.any(grid < 1)
        or not np.isfinite(cell).all()
        or not 0 < fraction <= 1
    ):
        raise ValueError("Invalid ACE dimensions or physical metadata")
    blocks = {name: [] for name in block_names}
    points, mappings = [], []
    for k in range(nk):
        index, ng = map(int, take("<i4", 2))
        point = take("<f8", 3)
        if index != k + 1 or not 0 < ng <= nr or not np.isfinite(point).all():
            raise ValueError("Invalid ACE k-point record")
        mapping = take("<i4", ng)
        if (
            len(np.unique(mapping)) != ng
            or np.any(mapping < 1)
            or np.any(mapping > math.prod(map(int, grid)))
        ):
            raise ValueError("Invalid ACE reciprocal mapping")
        for name in block_names:
            block = take("<c16", ng * nb).reshape(ng, nb, order="F")
            if not np.isfinite(block).all():
                raise ValueError("Nonfinite exchange coefficients")
            blocks[name].append(block)
        points.append(point)
        mappings.append(mapping)
    if offset != len(data):
        raise ValueError("Unexpected trailing ACE data")
    archive = dict(
        dims=dims,
        grid=grid,
        cell_A=cell,
        fraction=fraction,
        kpoints=np.array(points),
        indices=mappings,
        **blocks,
    )
    _validate_transport(archive, block_names)
    return archive


def _validate_transport(archive, block_names):
    dims, grid = np.asarray(archive["dims"]), np.asarray(archive["grid"])
    if dims.shape != (4,) or grid.shape != (3,):
        raise ValueError("Invalid exchange dimension shapes")
    for values in (dims, grid):
        if (
            not np.isfinite(values).all()
            or np.any(values < 1)
            or np.any(values > np.iinfo(np.int32).max)
            or not np.array_equal(values, values.astype(np.int64))
        ):
            raise ValueError("Exchange dimensions must be positive int32 values")
    nr, nb, nk, ns = map(int, dims)
    cell, points = np.asarray(archive["cell_A"]), np.asarray(archive["kpoints"])
    if (
        ns != 1
        or nb > nr
        or cell.shape != (3, 3)
        or points.shape != (nk, 3)
        or not np.isfinite(cell).all()
        or not np.isfinite(points).all()
        or np.linalg.det(cell) <= 0
        or not 0 < archive["fraction"] <= 1
    ):
        raise ValueError("Invalid scalar exchange metadata")
    if any(len(archive[name]) != nk for name in ("indices", *block_names)):
        raise ValueError("Exchange k-point block count mismatch")
    for k, mapping in enumerate(archive["indices"]):
        mapping = np.asarray(mapping)
        if (
            mapping.ndim != 1
            or not nb <= len(mapping) <= nr
            or not np.isfinite(mapping).all()
            or np.any(mapping < 1)
            or np.any(mapping > math.prod(map(int, grid)))
            or np.any(mapping > np.iinfo(np.int32).max)
            or not np.array_equal(mapping, mapping.astype(np.int64))
            or len(np.unique(mapping)) != len(mapping)
        ):
            raise ValueError("Invalid exchange reciprocal mapping")
        for name in block_names:
            block = np.asarray(archive[name][k])
            if block.shape != (len(mapping), nb) or not np.isfinite(block).all():
                raise ValueError(f"Invalid exchange {name} block")


def _write_transport(path, archive, magic, block_names):
    _validate_transport(archive, block_names)
    with Path(path).open("xb") as f:
        f.write(magic)
        for array, dtype in [
            (archive["dims"], "<i4"),
            (archive["grid"], "<i4"),
            (archive["cell_A"].T, "<f8"),
            ([archive["fraction"]], "<f8"),
        ]:
            f.write(np.asarray(array, dtype=dtype).tobytes(order="F"))
        for k, (point, mapping) in enumerate(
            zip(archive["kpoints"], archive["indices"])
        ):
            f.write(np.array([k + 1, len(mapping)], dtype="<i4").tobytes())
            f.write(np.asarray(point, dtype="<f8").tobytes())
            f.write(np.asarray(mapping, dtype="<i4").tobytes())
            for name in block_names:
                f.write(np.asarray(archive[name][k], dtype="<c16").tobytes(order="F"))


def read_ace(path):
    return _read_transport(path, MAGIC, ("factors",))


def write_ace(path, archive, *, scale=1.0):
    """Serialize factors; scale changes the exchange action, not its sign."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("ACE scale must be finite and nonnegative")
    _validate_transport(archive, ("factors",))
    scaled = dict(
        archive,
        factors=[np.asarray(block) * np.sqrt(scale) for block in archive["factors"]],
    )
    _write_transport(path, scaled, MAGIC, ("factors",))


def read_exchange_samples(path):
    return _read_transport(path, SAMPLES_MAGIC, ("orbitals", "actions"))


def write_exchange_samples(path, archive):
    _write_transport(path, archive, SAMPLES_MAGIC, ("orbitals", "actions"))


def apply_ace(archive, k, coefficients):
    x = archive["factors"][k]
    c = np.asarray(coefficients)
    if c.ndim not in (1, 2) or c.shape[0] != len(x) or not np.isfinite(c).all():
        raise ValueError("ACE coefficients must use the stored native reciprocal basis")
    return -x @ (x.conj().T @ c)


def export_hybrid(reference, output):
    from .exchange import build_ace_from_samples

    reference, output = Path(reference).resolve(), Path(output).resolve()
    m = json.loads((reference / "manifest.json").read_text())
    settings = m["settings"]
    _require_pbe0(settings)
    if m.get("functional") != "PBE0" or settings.get("ALGO") != "Normal":
        raise ValueError("Exchange export requires a scalar PBE0 ALGO=Normal reference")
    if (
        "full nonlocal exchange samples exported before compression"
        not in (reference / "stdout.log").read_text()
    ):
        raise ValueError("Full exchange sample export marker missing")
    samples_path = reference / "MAPHO_FOCK.bin"
    samples = read_exchange_samples(samples_path)
    ace = samples
    bands = read_bands(reference)
    if bands["eigenvalues_eV"].shape != (int(ace["dims"][2]), int(ace["dims"][1])):
        raise ValueError("ACE construction rank/mesh differs from reference bands")
    if not np.allclose(ace["kpoints"], bands["kpoints"], atol=1e-10, rtol=0):
        raise ValueError("ACE and native band k points differ")
    components = output.with_name(output.stem + "_components.h5")
    ace_path = output.with_suffix(".ace.bin")
    if any(path.exists() for path in (output, components, ace_path)):
        raise FileExistsError("Hybrid export would overwrite an existing artifact")
    if abs(samples["fraction"] - float(settings.get("AEXX", 0.25))) > 1e-12:
        raise ValueError("Exchange fraction differs from reference")
    factorization = build_ace_from_samples(samples_path, ace_path)
    ace = read_ace(ace_path)
    meta = export_operator(reference, components, components_only=True)
    with h5py.File(components) as f:
        if not np.allclose(f["cell_A"][:], ace["cell_A"], atol=1e-10, rtol=0):
            raise ValueError("ACE and component cells differ")
    metadata = dict(
        schema=SCHEMA,
        components=str(components),
        components_sha256=sha(components),
        ace_file=str(ace_path),
        ace_sha256=sha(ace_path),
        factorization=factorization,
        reference=str(reference),
        reference_manifest=m,
        representation="MAPHO ACE from full nonlocal Fock actions plus smooth potential and PAW D",
        complete_operator_components=True,
        exact_arbitrary_fock_action=False,
        exactness_scope="ACE construction subspace; same mesh and plane-wave basis only",
        rank_per_k=int(ace["dims"][1]),
        component_metadata=meta,
    )
    with h5py.File(output, "x") as f:
        f.attrs["schema"] = SCHEMA
        f.attrs["metadata_json"] = json.dumps(metadata)
        f["kpoints"] = ace["kpoints"]
        f["cell_A"] = ace["cell_A"]
        for k, (x, mapping) in enumerate(zip(ace["factors"], ace["indices"])):
            f.create_dataset(f"ace/{k}/factors", data=x, compression="gzip")
            f[f"ace/{k}/indices"] = mapping
    return metadata


def _require_pbe0(settings):
    hybrid = settings.get("LHFCALC") is True or str(
        settings.get("LHFCALC")
    ).upper() in ("TRUE", ".TRUE.", "T")
    if (
        settings.get("ISPIN") != 1
        or not hybrid
        or float(settings.get("HFSCREEN", 0)) != 0
        or float(settings.get("HFRCUT", 0)) != 0
        or float(settings.get("AEXX", 0.25)) != 0.25
    ):
        raise ValueError(
            "Interface requires scalar unscreened PBE0 with AEXX=0.25 and HFRCUT=0"
        )


def prepare_hybrid_replay(
    operator, folder, *, potential=None, d=None, shift_eV=0.0, ace_scale=1.0
):
    operator, folder = Path(operator).resolve(), Path(folder).resolve()
    if not np.isfinite(ace_scale) or ace_scale < 0:
        raise ValueError("ACE scale must be finite and nonnegative")
    with h5py.File(operator) as f:
        if f.attrs["schema"] != SCHEMA:
            raise ValueError("Expected a native PAW+ACE archive")
        meta = json.loads(f.attrs["metadata_json"])
    _require_pbe0(meta["reference_manifest"]["settings"])
    for key in ("components", "ace"):
        path = meta["components" if key == "components" else "ace_file"]
        if sha(path) != meta[key + "_sha256"]:
            raise ValueError(f"Archived {key} changed")
    ace = read_ace(meta["ace_file"])
    m = prepare_replay(
        meta["components"],
        folder,
        potential=potential,
        d=d,
        shift_eV=shift_eV,
        omit_fock=True,
    )
    s = m["settings"]
    s.pop("TIME", None)
    s.update(
        LHFCALC=True,
        LFOCKACE=True,
        ALGO="Normal",
        AEXX=ace["fraction"],
        HFSCREEN=0.0,
        HFRCUT=0.0,
        PRECFOCK="Accurate",
        NELMIN=40,
        NELM=160,
        EDIFF=1e-10,
        ISTART=0,
        ICHARG=14,
        NELMDL=0,
        LWAVE=False,
        LCHARG=False,
    )
    write_incar(folder / "INCAR", s)
    write_ace(folder / "MAPHO_ACE.bin", ace, scale=ace_scale)
    (folder / "MAPHO_ACE_IMPORT").touch()
    m.pop("omitted_components", None)
    m.pop("diagnostic", None)
    m.pop("complete_hamiltonian", None)
    m.update(
        schema="mapho.vasp_paw_ace_replay.v2",
        complete_operator_components=True,
        representation=meta["representation"],
        exact_arbitrary_fock_action=False,
        exchange_application="private MAPHO contraction; native ACE contraction bypassed",
        ace_scale=ace_scale,
        operator=str(operator),
        operator_sha256=sha(operator),
        inputs={
            name: sha(folder / name)
            for name in (
                "POSCAR",
                "POTCAR",
                "KPOINTS",
                "INCAR",
                "POT",
                "MLHAM_PAW.dat",
                "MAPHO_ACE.bin",
                "MAPHO_ACE_IMPORT",
                "MLHAM_IMPORT",
            )
        },
    )
    save_json(folder / "manifest.json", m)
    return m
