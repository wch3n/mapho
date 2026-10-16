"""Lazy aliases for historical imports; all implementations live in mapho.

Keep module and class identity intact for old checkpoints. This namespace
contains no model implementation and imports no model until it is requested.
"""

import importlib
import importlib.abc
import importlib.util
import sys

from mapho import __version__


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Replace the placeholder, rather than changing the canonical module's
        # __name__/__spec__ or executing its definitions a second time.
        sys.modules[module.__name__] = importlib.import_module(self.target)

    def get_code(self, fullname):
        # runpy uses get_code for `python -m mlham[.module]`.
        return compile(
            "import runpy\nrunpy.run_module("
            + repr(self.target)
            + ', run_name="__main__")\n',
            "<legacy MAPHO command>",
            "exec",
        )


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        canonical = "mapho" + fullname[len(__name__) :]
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )


if not any(
    type(f).__module__ == __name__ and type(f).__name__ == "_AliasFinder"
    for f in sys.meta_path
):
    sys.meta_path.insert(0, _AliasFinder())
