"""Raw localized-Hamiltonian adapters. File conversion is not gauge alignment."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def read_blocks_h5(path):
    """Read [Rx,Ry,Rz,i,j] blocks from flat HDF5 arrays; retain complex data."""
    with h5py.File(path, "r") as source:
        pairs = source["atom_pairs"][:]
        bounds = source["chunk_boundaries"][:]
        shapes = source["chunk_shapes"][:]
        entries = source["entries"][:]
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 5
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise ValueError("atom_pairs must be an integer array of shape (N, 5)")
    if bounds.shape != (len(pairs) + 1,) or shapes.shape != (len(pairs), 2):
        raise ValueError("Invalid block metadata dimensions")
    if not np.issubdtype(bounds.dtype, np.integer) or not np.issubdtype(
        shapes.dtype, np.integer
    ):
        raise ValueError("Block boundaries and shapes must be integers")
    if entries.ndim != 1 or bounds[0] != 0 or bounds[-1] != len(entries):
        raise ValueError("Invalid entries or chunk boundaries")
    if (shapes <= 0).any() or not np.array_equal(np.diff(bounds), shapes.prod(axis=1)):
        raise ValueError("Block sizes do not match chunk boundaries")
    if (pairs[:, 3:] < 0).any() or not np.isfinite(entries).all():
        raise ValueError("Invalid atom indices or nonfinite entries")
    blocks = {}
    for row, start, end, shape in zip(pairs, bounds[:-1], bounds[1:], shapes):
        key = tuple(map(int, row))
        if key in blocks:
            raise ValueError("Duplicate atom-pair block")
        blocks[key] = entries[start:end].reshape(shape)
    return blocks


def write_blocks_h5(path, blocks):
    if not blocks:
        raise ValueError("At least one block is required")
    keys = sorted(blocks)
    matrices = [np.asarray(blocks[key]) for key in keys]
    if any(
        m.ndim != 2 or min(m.shape) < 1 or not np.isfinite(m).all() for m in matrices
    ):
        raise ValueError("Blocks must be finite nonempty matrices")
    pairs = np.asarray(keys)
    if (
        pairs.shape != (len(keys), 5)
        or not np.issubdtype(pairs.dtype, np.integer)
        or (pairs[:, 3:] < 0).any()
    ):
        raise ValueError("Block keys must be integer (Rx, Ry, Rz, i, j) tuples")
    with h5py.File(path, "x") as destination:
        destination["atom_pairs"] = pairs
        destination["chunk_shapes"] = np.asarray(
            [m.shape for m in matrices], dtype=np.int64
        )
        destination["chunk_boundaries"] = np.r_[
            0, np.cumsum([m.size for m in matrices])
        ]
        destination["entries"] = np.concatenate([m.ravel() for m in matrices])


@dataclass
class WannierHamiltonian:
    translations: np.ndarray
    matrices: np.ndarray  # degeneracy weights already applied, in eV
    degeneracies: np.ndarray

    def bloch(self, kpoints):
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3 or not np.isfinite(kpoints).all():
            raise ValueError("kpoints must have finite shape (Nk, 3)")
        phases = np.exp(2j * np.pi * kpoints @ self.translations.T)
        return np.einsum("kr,rij->kij", phases, self.matrices)


def read_wannier90_hr(path):
    """Read seedname_hr.dat, including Wigner-Seitz degeneracy factors.

    This is a raw matrix reader, not a training-sample converter. Arbitrary
    Wannier functions need not transform as atom-centred s/p/d orbitals.
    Orbital-dependent wsvec translations are not represented by hr.dat alone.
    """
    path = Path(path)
    with path.open() as stream:
        stream.readline()  # creation comment
        try:
            nw = int(stream.readline())
            nr = int(stream.readline())
            if nw < 1 or nr < 1:
                raise ValueError("Invalid Wannier matrix dimensions")
            degeneracies = []
            while len(degeneracies) < nr:
                line = stream.readline()
                if not line:
                    raise ValueError("Truncated degeneracy table")
                degeneracies.extend(map(int, line.split()))
            if len(degeneracies) != nr or min(degeneracies) < 1:
                raise ValueError("Invalid degeneracy table")
            matrices, translations, seen_r = [], [], set()
            for weight in degeneracies:
                matrix = np.zeros((nw, nw), dtype=complex)
                seen_elements, translation = set(), None
                for _ in range(nw * nw):
                    columns = stream.readline().split()
                    if len(columns) != 7:
                        raise ValueError("Truncated or malformed Hamiltonian record")
                    r = tuple(map(int, columns[:3]))
                    m, n = (int(c) - 1 for c in columns[3:5])
                    if translation is None:
                        translation = r
                    if (
                        r != translation
                        or not (0 <= m < nw and 0 <= n < nw)
                        or (m, n) in seen_elements
                    ):
                        raise ValueError(
                            "Invalid or duplicate Hamiltonian matrix element"
                        )
                    seen_elements.add((m, n))
                    real, imag = (
                        float(c.replace("D", "E").replace("d", "e"))
                        for c in columns[5:]
                    )
                    matrix[m, n] = complex(real, imag) / weight
                if translation in seen_r:
                    raise ValueError("Duplicate lattice translation")
                seen_r.add(translation)
                translations.append(translation)
                matrices.append(matrix)
            if any(line.strip() for line in stream):
                raise ValueError("Unexpected trailing Hamiltonian records")
        except (TypeError, IndexError) as error:
            raise ValueError("Malformed Wannier Hamiltonian") from error
    matrices = np.asarray(matrices)
    if not np.isfinite(matrices).all():
        raise ValueError("Nonfinite Wannier Hamiltonian")
    lookup = {r: h for r, h in zip(translations, matrices)}
    for r, h in lookup.items():
        reverse = lookup.get(tuple(-v for v in r))
        if reverse is None or not np.allclose(
            h, reverse.conj().T, atol=1e-7, rtol=1e-6
        ):
            raise ValueError("Wannier Hamiltonian violates H(R) = H(-R)^dagger")
    return WannierHamiltonian(
        np.asarray(translations), matrices, np.asarray(degeneracies)
    )
