#!/usr/bin/env python3
"""Equal-author endpoint loss equals average plus half-difference loss."""

import numpy as np
import pytest

from scripts.report_grain_full_context_geometry import paired_channel_account


def test_parallelogram_and_equal_author_weights():
    r = np.random.default_rng(6501)
    a = [r.normal(size=(t, 3)) for t in (10, 100)]
    b = [x + r.normal(size=x.shape) / 10 for x in a]
    p = [x * 0.8 for x in a]
    q = [x * 0.9 for x in b]
    d = paired_channel_account(a, b, p, q, 2.0)
    expected = np.mean(
        [
            0.5 * (np.mean((x - z) ** 2) + np.mean((y - w) ** 2)) / 4
            for x, y, z, w in zip(a, b, p, q)
        ]
    )
    assert d["pair_loss"] == pytest.approx(expected)
    assert d["parallelogram_error"] < 1e-12


def test_exact_average_with_missing_small_change_has_small_global_loss():
    mean = np.ones((20, 3))
    delta = np.full((20, 3), 0.01)
    d = paired_channel_account([mean - delta], [mean + delta], [mean], [mean], 1.0)
    assert d["average_loss"] == 0
    assert d["half_difference_sse_over_zero"] == pytest.approx(1.0)
    assert d["target_difference_fraction"] == pytest.approx(0.0001 / 1.0001)
    assert d["pair_loss"] == pytest.approx(0.0001)
