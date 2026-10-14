#!/usr/bin/env python3
"""The companion changes interface typing only, not scientific primitives."""

import numpy as np
import pytest

from scripts.complete_grain_recursive_collector_hm112 import current_arrays


def test_identical_arrays_not_recomputed():
    """Preserve exact references and byte values for both original channels."""
    k = np.arange(12).reshape(3, 2, 2)
    v = k + 4
    a, b = current_arrays({"k": k, "v": v})
    assert a is k and b is v


def test_wrong_interface_not_zero_filled():
    """An incomplete record cannot be silently converted to a valid observation."""
    with pytest.raises(ValueError):
        current_arrays({"k": np.zeros(3)})
