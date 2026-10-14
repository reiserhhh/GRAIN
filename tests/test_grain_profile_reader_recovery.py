#!/usr/bin/env python3
"""Binary storage compatibility with no rounding or reinterpretation."""

import numpy as np
import pytest

from scripts.recover_grain_profile_reader_hm182 import canonical_binary


def test_float_binary_preserves_values_and_returns_integer_codes():
    """Cover the actual reporting failure,not only synthetic int labels."""
    y = np.array([[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    result = canonical_binary(y)
    assert np.array_equal(y, result) and result.dtype == np.uint8


@pytest.mark.parametrize("bad", [0.1, np.nan])
def test_nonbinary_values_cannot_be_rounded(bad):
    """Reject semantic changes disguised as datatype repairs."""
    y = np.zeros((2, 5))
    y[0, 0] = bad
    with pytest.raises(ValueError):
        canonical_binary(y)
