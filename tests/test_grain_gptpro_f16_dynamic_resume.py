#!/usr/bin/env python3
"""Keep comparison dtype repair separate from the frozen scientific scorer."""

import numpy as np
import pytest

from scripts.complete_grain_gptpro_f16_dynamic_text import json_difference


def test_numpy_interval_roundoff_uses_numeric_tolerance():
    assert (
        json_difference({"lower": np.float64(0.1)}, {"lower": np.float64(0.1 + 2e-16)})
        < 1e-10
    )
    assert (
        json_difference({"lower": np.float64(0.1)}, {"lower": np.float64(0.2)}) > 0.09
    )


def test_non_numeric_mismatch_still_rejected():
    with pytest.raises(ValueError):
        json_difference({"status": "observed"}, {"status": "borrowed"})
