"""Release-boundary tests use synthetic private files, never licensed code."""

import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_check", ROOT / "tools/check_release.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize(
    "name",
    [
        "mapho/vasp_patch.py",
        "mapho/vasp_ace_patch.py",
        "mapho/vasp_bridge.F.inc",
        "mapho/vasp_ace_bridge.F.inc",
        "mapho/vasp/vasp_patch.py",
        "mapho/vasp/vasp_ace_patch.py",
        "mapho/vasp/vasp_bridge.F.inc",
        "mapho/vasp/vasp_ace_bridge.F.inc",
        "release/runs/private/file.txt",
        "release/POTCAR",
        "release/POTCAR.Si",
        "release/vasp_std",
        "release/source.F",
        "../outside.py",
    ],
)
def test_private_names_rejected(name):
    assert checker.forbidden(name)


def test_public_names_allowed():
    for name in ["mapho/ace_operator.py", "mapho/vasp/paw.py", "mapho/file_utils.py"]:
        assert not checker.forbidden(name)


def test_archive_checker_fails_on_private_members_and_links(tmp_path):
    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mapho/ace_operator.py", "# synthetic public file")
        archive.writestr("mapho/vasp_ace_patch.py", "# synthetic private sentinel")
    assert checker.check_archive(wheel) == ["mapho/vasp_ace_patch.py"]
    source = tmp_path / "bad.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        entry = tarfile.TarInfo("release/POTCAR")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
        link = tarfile.TarInfo("release/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/private"
        archive.addfile(link)
    assert checker.check_archive(source) == ["release/POTCAR", "release/link"]
    # A VASP package directory is allowed; a binary with that name is not.
    assert not checker.forbidden("mapho/vasp", is_directory=True)
    assert checker.forbidden("mapho/vasp", is_directory=False)


def build(folder, operation):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f'from setuptools import build_meta; build_meta.{operation}("dist")',
        ],
        cwd=folder,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout


def test_wheel_sdist_and_rebuilt_wheel_exclude_private_files(tmp_path):
    project = tmp_path / "project"
    package = project / "src/mapho"
    package.mkdir(parents=True)
    for name in ["setup.py", "pyproject.toml", "MANIFEST.in"]:
        shutil.copy2(ROOT / name, project / name)
    (project / "README.md").write_text("Synthetic packaging fixture.\n")
    for name in [
        "__init__.py",
        "ace_operator.py",
        "file_utils.py",
        "vasp/__init__.py",
        "vasp/paw.py",
        "qe/__init__.py",
        "qe/backend.py",
    ]:
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text("# synthetic public module\n")
    legacy = project / "src/mlham"
    legacy.mkdir()
    (legacy / "__init__.py").write_text("# synthetic compatibility shim\n")
    private = [
        "vasp_patch.py",
        "vasp_ace_patch.py",
        "vasp_bridge.F.inc",
        "vasp_ace_bridge.F.inc",
    ]
    private += ["vasp/" + name for name in private]
    for name in private:
        (package / name).write_text("SYNTHETIC_PRIVATE_SENTINEL\n")
    stale = project / "build/lib/mapho"
    stale.mkdir(parents=True)
    for name in private + ["qe.py", "vasp_paw.py"]:
        (stale / name).parent.mkdir(parents=True, exist_ok=True)
        (stale / name).write_text("SYNTHETIC_STALE_SENTINEL\n")
    legacy_stale = project / "build/lib/mlham"
    legacy_stale.mkdir()
    (legacy_stale / "qe.py").write_text("SYNTHETIC_OBSOLETE_WRAPPER\n")
    run = project / "runs/private"
    run.mkdir(parents=True)
    (run / "POTCAR").write_text("SYNTHETIC_DATA_SENTINEL\n")

    build(project, "build_wheel")
    build(project, "build_sdist")
    wheel = next((project / "dist").glob("*.whl"))
    source = next((project / "dist").glob("*.tar.gz"))
    for archive in [wheel, source]:
        assert checker.check_archive(archive) == []
    with zipfile.ZipFile(wheel) as archive:
        assert "mapho/ace_operator.py" in archive.namelist()
        assert "mapho/vasp/paw.py" in archive.namelist()
        assert "mapho/qe/backend.py" in archive.namelist()
        assert "mapho/qe.py" not in archive.namelist()
        assert "mapho/vasp_paw.py" not in archive.namelist()
        assert [name for name in archive.namelist() if name.startswith("mlham/")] == [
            "mlham/__init__.py"
        ]
    with tarfile.open(source) as archive:
        assert any(name.endswith("/setup.py") for name in archive.getnames())
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    build(unpacked, "build_wheel")
    assert checker.check_archive(next((unpacked / "dist").glob("*.whl"))) == []
    # Exclusion must not remove the local source files needed by licensed users.
    assert all((package / name).is_file() for name in private)
