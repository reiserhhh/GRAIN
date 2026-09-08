#!/usr/bin/env python3
"""Paired participant contrasts and descriptive covariance accounting."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_grain_counseling_views import common_specific, paired_delta


def test_identical_view_has_exact_zero_delta():
    x = np.arange(20.0)
    y = x + np.sin(x)
    result = paired_delta(x, x, y, draws=50)
    assert result["delta_r_free_minus_direct"] == 0
    assert result["delta_lower"] == result["delta_upper"] == 0
    with pytest.raises(ValueError, match="Matched"):
        paired_delta(x, x[:-1], y)


def test_signed_common_and_specific_reconstruct_association():
    rng = np.random.default_rng(41)
    common = rng.normal(size=(60, 1))
    scores = pd.DataFrame(common + rng.normal(size=(60, 4)), columns=list("abcd"))
    labels = scores + rng.normal(size=(60, 4))
    defs = {q: {"family": "same" if q in "ab" else q, "phase": "pre"} for q in scores}
    geometry, result = common_specific(scores, defs, labels)
    assert geometry["participants"] == 60
    assert result.accounting_error.max() < 1e-12
    assert list(result.other_query_count) == [2, 2, 3, 3]
    with pytest.raises(ValueError, match="alignment"):
        common_specific(scores, defs, labels.iloc[::-1])
