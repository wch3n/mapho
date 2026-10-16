"""Self-contained public Python snapshots for reproducible calculation workers."""

from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import shutil

from .file_utils import sha


def package_root():
    return Path(__file__).resolve().parent


def source_path(name):
    """Resolve a manifest path relative to the active MAPHO package."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid source path: {name}")
    return package_root().joinpath(*relative.parts)


def source_hashes(directory=None):
    """Hash public Python modules, preserving the package directory structure."""
    directory = package_root() if directory is None else Path(directory)
    return {
        p.relative_to(directory).as_posix(): sha(p)
        for p in sorted(directory.rglob("*.py"))
        if "__pycache__" not in p.parts and not fnmatchcase(p.name, "vasp*_patch.py")
    }


def snapshot_package(destination):
    """Copy public modules to an empty package directory and return their hashes.

    Private patchers and Fortran bridges are unnecessary for calculation workers
    and are excluded. Existing snapshots are never overwritten.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Source snapshot is not empty: {destination}")
    hashes = source_hashes()
    for name, expected in hashes.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(name), target)
        if sha(target) != expected:
            raise RuntimeError(f"Source changed while copying: {name}")
    return hashes
