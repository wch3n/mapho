"""Keep local VASP patching material out of binary distributions."""

from fnmatch import fnmatchcase
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


def private_file(name):
    return (
        fnmatchcase(name, "vasp*_patch.py")
        or fnmatchcase(name, "*_bridge.F.inc")
        or Path(name).suffix.lower() in {".f", ".f90", ".inc"}
    )


class PublicBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            entry
            for entry in super().find_package_modules(package, package_dir)
            if not private_file(Path(entry[2]).name)
        ]

    def run(self):
        super().run()
        # Prune modules left in build/lib by earlier flat layouts. A wheel
        # must contain only current source files, including nested packages.
        for package in ("mapho", "mlham"):
            built = Path(self.build_lib) / package
            source = Path(self.get_package_dir(package))
            for path in built.rglob("*.py"):
                if not (source / path.relative_to(built)).is_file():
                    path.unlink()
        # A wheel must not pick up excluded files left by an earlier build.
        for path in Path(self.build_lib).rglob("*"):
            if path.is_file() and private_file(path.name):
                path.unlink()


setup(cmdclass={"build_py": PublicBuildPy})
