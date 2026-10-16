"""Load e3nn constants with the serialization defaults of recent PyTorch."""

from contextlib import nullcontext

import torch

# e3nn 0.4.4 stores built-in slice objects alongside its Wigner tensors.
# PyTorch >=2.6 defaults to weights-only loading. Allow just that built-in
# during import, without changing torch.load or the process-wide defaults.
safe_globals = getattr(torch.serialization, "safe_globals", None)
existing = getattr(torch.serialization, "get_safe_globals", lambda: [])()
additional = [] if slice in existing else [slice]
with safe_globals(additional) if safe_globals is not None else nullcontext():
    from e3nn import o3
    from e3nn.nn import NormActivation

__all__ = ["o3", "NormActivation"]
