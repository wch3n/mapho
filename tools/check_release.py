#!/usr/bin/env python3
"""Reject known private material in wheels and source archives; never extract it."""

import argparse
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


PRIVATE_DIRECTORIES = {
    "runs",
    "build",
    ".git",
    ".venv",
    ".venv-qe",
    "__pycache__",
    ".pytest_cache",
    "build_mapho_report",
}
PRIVATE_PATTERNS = (
    "vasp*_patch.py",
    "*_bridge.F.inc",
    "*.f",
    "*.F",
    "*.f90",
    "*.F90",
    "*.inc",
    "*.pyc",
    "*.pyo",
    "POTCAR",
    "POTCAR.*",
    "*.UPF",
    "*.upf",
    "WAVECAR",
    "CHGCAR",
    "vasp",
    "vasp_std",
    "vasp_gam",
    "vasp_ncl",
)


def forbidden(name, *, is_directory=False):
    path = PurePosixPath(name)
    return (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or bool(PRIVATE_DIRECTORIES.intersection(path.parts))
        or (
            not is_directory
            and any(fnmatchcase(path.name, pattern) for pattern in PRIVATE_PATTERNS)
        )
    )


def check_archive(path):
    """Return names of disallowed entries, without reading member contents."""
    rejected = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if forbidden(
                    entry.filename, is_directory=entry.is_dir()
                ) or stat.S_ISLNK(entry.external_attr >> 16):
                    rejected.append(entry.filename)
    else:
        with tarfile.open(path, "r:*") as archive:
            for entry in archive:
                if forbidden(entry.name, is_directory=entry.isdir()) or not (
                    entry.isfile() or entry.isdir()
                ):
                    rejected.append(entry.name)
    return sorted(set(rejected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.archives:
        rejected = check_archive(path)
        if rejected:
            failed = True
            print(f"FAIL {path}: forbidden entries: {rejected}")
        else:
            print(f"PASS {path}: no forbidden archive entries")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
