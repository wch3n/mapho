"""Lazy aliases for flat backend imports used by historical MAPHO checkpoints."""

import importlib
import importlib.abc
import importlib.util
import sys

# One implementation per module: old imports resolve to the canonical object.
BACKEND_ALIASES = {
    "mapho.ace_validation": "mapho.qe.ace_validation",
    "mapho.field_dataset": "mapho.qe.field_dataset",
    "mapho.kpoint_convergence": "mapho.qe.kpoint_convergence",
    "mapho.kpoint_recovery": "mapho.qe.kpoint_recovery",
    "mapho.qe_exchange": "mapho.qe.exchange",
    "mapho.qe_files": "mapho.qe.files",
    "mapho.supercell_reference": "mapho.qe.supercell_reference",
    "mapho.upf": "mapho.qe.upf",
    "mapho.vasp_ace": "mapho.vasp.ace",
    "mapho.vasp_ace_patch": "mapho.vasp.vasp_ace_patch",
    "mapho.vasp_ace_validation": "mapho.vasp.ace_validation",
    "mapho.vasp_activation_ablation": "mapho.vasp.activation_ablation",
    "mapho.vasp_atomic_reference": "mapho.vasp.atomic_reference",
    "mapho.vasp_binary_primitive_bands": "mapho.vasp.binary_primitive_bands",
    "mapho.vasp_binary_primitive_plot": "mapho.vasp.binary_primitive_plot",
    "mapho.vasp_exchange": "mapho.vasp.exchange",
    "mapho.vasp_hybrid_pilot": "mapho.vasp.hybrid_pilot",
    "mapho.vasp_material_dataset": "mapho.vasp.material_dataset",
    "mapho.vasp_material_training": "mapho.vasp.material_training",
    "mapho.vasp_model": "mapho.vasp.model",
    "mapho.vasp_operator_diagnostic": "mapho.vasp.operator_diagnostic",
    "mapho.vasp_patch": "mapho.vasp.vasp_patch",
    "mapho.vasp_paw": "mapho.vasp.paw",
    "mapho.vasp_paw_conditioning": "mapho.vasp.paw_conditioning",
    "mapho.vasp_paw_joint": "mapho.vasp.paw_joint",
    "mapho.vasp_paw_repair": "mapho.vasp.paw_repair",
    "mapho.vasp_paw_validation": "mapho.vasp.paw_validation",
    "mapho.vasp_primitive_bands": "mapho.vasp.primitive_bands",
    "mapho.vasp_primitive_plot": "mapho.vasp.primitive_plot",
    "mapho.vasp_reference_chunk": "mapho.vasp.reference_chunk",
    "mapho.vasp_silicon_dataset": "mapho.vasp.silicon_dataset",
    "mapho.vasp_spin_validation": "mapho.vasp.spin_validation",
    "mapho.vasp_train": "mapho.vasp.train",
    "mapho.vasp_training_validation": "mapho.vasp.training_validation",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[module.__name__] = importlib.import_module(self.target)

    def get_code(self, fullname):
        return compile(
            "import runpy\nrunpy.run_module("
            + repr(self.target)
            + ', run_name="__main__")\n',
            "<legacy MAPHO command>",
            "exec",
        )


class _BackendAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = BACKEND_ALIASES.get(fullname)
        if canonical is None:
            return None
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )


def install_backend_aliases():
    if not any(
        type(f).__module__ == __name__ and type(f).__name__ == "_BackendAliasFinder"
        for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _BackendAliasFinder())
