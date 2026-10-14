#!/usr/bin/env python3
"""Compatibility conversions preserve binary values and reject approximations."""

import numpy as np
import pytest

from scripts.run_grain_path_functional_dtype_hm236 import exact_binary_codes


def test_exact_float_binary_values_unchanged():
    """The serialized old float bits may be safely addressed as integers."""
    y = np.array([[0.0, 1.0, 1.0, 0.0, 1.0]])
    z = exact_binary_codes(y)
    assert z.dtype == np.int64
    np.testing.assert_array_equal(y, z)


@pytest.mark.parametrize("value", [0.999999, np.nan, 2.0])
def test_nonbinary_values_not_rounded(value):
    """The adapter must not repair malformed or nonfinite SOURCE targets."""
    with pytest.raises(ValueError):
        exact_binary_codes(np.array([0.0, value]))
