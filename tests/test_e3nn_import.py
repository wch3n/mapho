"""The dependency import must not relax serialization outside that import."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("already_allowed", [False, True])
def test_e3nn_import_preserves_slice_policy_and_loader(already_allowed):
    code = (
        f"""
import torch
if {already_allowed} and hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])
"""
        + """
original_load = torch.load
get_allowed = getattr(torch.serialization, 'get_safe_globals', lambda: [])
original_allowed = set(get_allowed())
from mapho._e3nn import o3
assert o3.wigner_3j(1, 1, 0).shape == (3, 3, 1)
assert torch.load is original_load
# PyTorch imports can register their own tensor/distributed classes.
assert original_allowed <= set(get_allowed())
assert (slice in get_allowed()) == (slice in original_allowed)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
