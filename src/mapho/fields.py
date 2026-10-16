"""Portable scalar electronic fields, independent of the orbital-block format."""

from .schema import canonical_schema

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


FIELD_UNITS = {
    "density": "electron/angstrom^3",
    "reference_density": "electron/angstrom^3",
    "ionic": "eV",
    "hartree": "eV",
    "xc": "eV",
    "reference_hxc": "eV",
}


def from_qe_grid(array, shape):
    """QE's first lattice index varies fastest; expose axes (i, j, k)."""
    array = np.asarray(array)
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected three positive grid dimensions")
    if array.shape != (int(np.prod(shape)), 1):
        raise ValueError("Only a serial scalar, unpolarized QE grid is supported")
    return array[:, 0].reshape(shape, order="F").copy()


def to_qe_grid(field):
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or not np.isfinite(field).all():
        raise ValueError("Expected a finite three-dimensional scalar field")
    return np.asfortranarray(field.reshape((-1, 1), order="F"))


@dataclass
class FieldSample:
    """Periodic grid without duplicated boundary points; cell vectors are rows."""

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    fields: dict
    kpoints: np.ndarray
    kweights: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray
    metadata: dict

    @property
    def shape(self):
        return self.fields["density"].shape

    @property
    def dv(self):
        return abs(np.linalg.det(self.cell)) / np.prod(self.shape)

    @property
    def hxc(self):
        return self.fields["hartree"] + self.fields["xc"]

    def residual(self):
        """Return zero-mean Hxc residual AND its offset; reconstruction needs both."""
        delta = self.hxc - self.fields["reference_hxc"]
        offset = float(delta.mean())
        return delta - offset, offset

    def validate(self):
        n = len(self.numbers)
        if (
            self.numbers.shape != (n,)
            or n == 0
            or not np.issubdtype(self.numbers.dtype, np.integer)
            or np.any(self.numbers < 1)
            or np.any(self.numbers > 118)
        ):
            raise ValueError("Invalid atomic numbers")
        if self.positions.shape != (n, 3) or self.cell.shape != (3, 3):
            raise ValueError("Invalid geometry dimensions")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.cell).all():
            raise ValueError("Non-finite geometry")
        if np.linalg.det(self.cell) <= 1e-10:
            raise ValueError("Expected a right-handed, nonzero periodic cell")
        if set(self.fields) != set(FIELD_UNITS):
            raise ValueError(f"Required fields: {sorted(FIELD_UNITS)}")
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Invalid grid shape")
        for name, value in self.fields.items():
            if value.shape != self.shape or not np.isfinite(value).all():
                raise ValueError(f"Inconsistent or non-finite field: {name}")
        nk = len(self.kweights)
        if self.kpoints.shape != (nk, 3) or nk == 0:
            raise ValueError("Invalid k-point dimensions")
        if (
            self.eigenvalues.ndim != 2
            or self.eigenvalues.shape[0] != nk
            or self.eigenvalues.shape[1] == 0
            or self.occupations.shape != self.eigenvalues.shape
        ):
            raise ValueError("Invalid eigenvalue or occupation dimensions")
        for value in (self.kpoints, self.kweights, self.eigenvalues, self.occupations):
            if not np.isfinite(value).all():
                raise ValueError("Non-finite electronic data")
        if np.any(self.kweights <= 0) or not np.isclose(
            self.kweights.sum(), 1, atol=1e-10
        ):
            raise ValueError("k-point weights must be positive and sum to one")
        if np.any(self.occupations < -1e-8) or np.any(self.occupations > 2 + 1e-8):
            raise ValueError("Spin-degenerate occupations must lie in [0, 2]")
        m = self.metadata
        if (
            canonical_schema(m.get("schema")) != "mapho.scalar_fields.v1"
            or m.get("field_units") != FIELD_UNITS
            or m.get("length_unit") != "angstrom"
            or m.get("energy_unit") != "eV"
        ):
            raise ValueError("Unsupported schema or units")
        if (
            m.get("grid_convention") != "fractional_ijk_endpoint_excluded"
            or m.get("grid_origin_fractional") != [0, 0, 0]
            or m.get("pbc") != [True, True, True]
            or m.get("nspin") != 1
        ):
            raise ValueError("Unsupported grid, boundary, or spin convention")
        if not m.get("potential_gauge") or not m.get("reference_definition"):
            raise ValueError("Potential gauge and reference definition are required")
        if not m.get("pseudopotentials"):
            raise ValueError("Pseudopotential provenance is required")
        nelect = float(m["nelect"])
        if not np.isfinite(nelect) or nelect <= 0:
            raise ValueError("Invalid electron count")
        for name in ("density", "reference_density"):
            if abs(self.fields[name].sum() * self.dv - nelect) > 1e-5:
                raise ValueError(f"Electron count mismatch in {name}")
        if abs(np.sum(self.occupations * self.kweights[:, None]) - nelect) > 1e-7:
            raise ValueError("Occupation electron count mismatch")

    def save(self, path):
        self.validate()
        arrays = {
            name: getattr(self, name)
            for name in (
                "numbers",
                "positions",
                "cell",
                "kpoints",
                "kweights",
                "eigenvalues",
                "occupations",
            )
        }
        arrays.update({f"field_{name}": value for name, value in self.fields.items()})
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=json.dumps(
                    dict(
                        self.metadata, schema=canonical_schema(self.metadata["schema"])
                    )
                ),
                **arrays,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                name: data[name]
                for name in (
                    "numbers",
                    "positions",
                    "cell",
                    "kpoints",
                    "kweights",
                    "eigenvalues",
                    "occupations",
                )
            }
            result = cls(
                **arrays,
                fields={name: data[f"field_{name}"] for name in FIELD_UNITS},
                metadata=json.loads(str(data["metadata"])),
            )
        result.validate()
        return result


@dataclass
class AtomicReference:
    """Inference-only QE initialization: no SCF labels or eigenvectors."""

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    fields: dict
    kpoints: np.ndarray
    metadata: dict

    @property
    def shape(self):
        return self.fields["reference_density"].shape

    @property
    def dv(self):
        return abs(np.linalg.det(self.cell)) / np.prod(self.shape)

    def validate(self):
        if (
            self.numbers.ndim != 1
            or not len(self.numbers)
            or not np.issubdtype(self.numbers.dtype, np.integer)
            or np.any(self.numbers < 1)
            or np.any(self.numbers > 118)
            or self.positions.shape != (len(self.numbers), 3)
            or self.cell.shape != (3, 3)
            or np.linalg.det(self.cell) <= 1e-10
            or self.kpoints.ndim != 2
            or self.kpoints.shape[1] != 3
            or not len(self.kpoints)
        ):
            raise ValueError("Invalid atomic reference geometry")
        if set(self.fields) != {"reference_density", "reference_hxc"}:
            raise ValueError("Atomic references contain only initial density and Hxc")
        if (
            len(self.shape) != 3
            or min(self.shape) < 1
            or any(a.shape != self.shape for a in self.fields.values())
        ):
            raise ValueError("Inconsistent atomic reference grids")
        if not all(
            np.isfinite(a).all()
            for a in (self.positions, self.cell, self.kpoints, *self.fields.values())
        ):
            raise ValueError("Non-finite atomic reference")
        m = self.metadata
        if (
            canonical_schema(m.get("schema")) != "mapho.atomic_reference.v1"
            or m.get("nspin") != 1
            or m.get("field_units") != {n: FIELD_UNITS[n] for n in self.fields}
            or m.get("length_unit") != "angstrom"
            or m.get("energy_unit") != "eV"
            or m.get("grid_convention") != "fractional_ijk_endpoint_excluded"
            or m.get("grid_origin_fractional") != [0, 0, 0]
            or m.get("pbc") != [True, True, True]
            or not all(
                m.get(k)
                for k in (
                    "potential_gauge",
                    "reference_definition",
                    "pseudopotentials",
                    "pseudo_paths",
                    "qe_config",
                )
            )
        ):
            raise ValueError("Unsupported atomic reference metadata")
        nelect = m.get("nelect", float("nan"))
        if (
            not np.isfinite(nelect)
            or nelect <= 0
            or abs(self.fields["reference_density"].sum() * self.dv - nelect) > 1e-5
        ):
            raise ValueError("Atomic reference electron count mismatch")

    def save(self, path):
        self.validate()
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                numbers=self.numbers,
                positions=self.positions,
                cell=self.cell,
                kpoints=self.kpoints,
                metadata=json.dumps(
                    dict(
                        self.metadata, schema=canonical_schema(self.metadata["schema"])
                    )
                ),
                **self.fields,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                *(data[k] for k in ("numbers", "positions", "cell")),
                fields={k: data[k] for k in ("reference_density", "reference_hxc")},
                kpoints=data["kpoints"],
                metadata=json.loads(str(data["metadata"])),
            )
        result.validate()
        return result


def load_operator_reference(path):
    with np.load(path, allow_pickle=False) as data:
        schema = json.loads(str(data["metadata"]))["schema"]
    if canonical_schema(schema) == "mapho.scalar_fields.v1":
        return FieldSample.load(path)
    if canonical_schema(schema) == "mapho.atomic_reference.v1":
        return AtomicReference.load(path)
    raise ValueError("Unknown operator reference schema")
