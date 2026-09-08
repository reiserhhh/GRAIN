#!/usr/bin/env python3
"""The same scores must not acquire a paired difference in either metric."""

import numpy as np
import pytest

from scripts.analyze_grain_all_factor_source import paired_quality


def test_same_source_predictor_has_zero_difference():
    x = np.arange(20.0) / 19
    y = (x > 0.5).astype(int)
    r = paired_quality(x, x, y, binary=True, draws=50)
    for name in ("delta_r", "delta_auc"):
        assert r[name] == r[name + "_lower"] == r[name + "_upper"] == 0
    assert r["function_auc"] == 1
    with pytest.raises(ValueError, match="Aligned"):
        paired_quality(x, x[:-1], y)
