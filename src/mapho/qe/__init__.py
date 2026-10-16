"""Quantum ESPRESSO interfaces, reference generation, and validation.

Historical ``from mapho.qe import QEConfig`` calls remain supported. New code
can import the implementation explicitly from ``mapho.qe.backend``.
"""


def __getattr__(name):
    from importlib import import_module

    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(import_module(".backend", __name__), name)
