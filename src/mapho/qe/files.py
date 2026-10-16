"""Resolve native restart paths from the prefix of the prepared QE input."""

from pathlib import Path
import re

from ase.io.espresso import read_fortran_namelist


def save_directory(calculation):
    """Support historical prefixes without guessing which save is authoritative."""
    calculation = Path(calculation)
    with (calculation / "qe.in").open() as stream:
        parameters, _ = read_fortran_namelist(stream)
    prefix = parameters.get("control", {}).get("prefix")
    if not isinstance(prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]*", prefix
    ):
        raise ValueError("Expected an explicit QE prefix without directory components")
    return calculation / "scratch" / (prefix + ".save")
