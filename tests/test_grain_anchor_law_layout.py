#!/usr/bin/env python3
"""Literal parent-cache addressing and contact-axis regression tests."""

import numpy as np
import pytest

from grain_research.construction import math_anchor_law_layout as M
from scripts.run_grain_anchor_law_disintegration_hm217 import contacts


def test_advanced_index_cache_is_literal_same_readers():
    raw = np.arange(24 * 2 * 12 * 3 * 5 * 4).reshape(24, 2, 12, 3, 5, 4)
    wrong = raw[:, :, [0, 6], 2, :, 2]
    restored = M.restore(wrong)
    expected = raw[:, :, [0, 6]][:, :, :, 2, :, 2]
    np.testing.assert_array_equal(restored, expected)
    for u, v, reader, q in np.ndindex(24, 2, 2, 5):
        assert restored[u, v, reader, q] == raw[u, v, (0, 6)[reader], 2, q, 2]


def test_contact_shape_and_view_mean_match_direct_covariance():
    rng = np.random.default_rng(217)
    x = rng.normal(size=(2, 24, 2, 3, 3))
    y = M.restore(rng.normal(size=(2, 24, 2, 5)))
    result = contacts(x, y)
    assert result.shape == (2, 3, 3, 2, 3, 5)
    for p, law, part, reader, view, q in np.ndindex(result.shape):
        a = x[p, :, view, law, part] if view < 2 else x[p, :, :, law, part].mean(1)
        b = y[:, view, reader, q] if view < 2 else y[:, :, reader, q].mean(1)
        np.testing.assert_allclose(
            result[p, law, part, reader, view, q], np.cov(a, b)[0, 1], atol=1e-14
        )


def test_unregistered_layout_fails():
    with pytest.raises(ValueError):
        M.restore(np.ones((24, 2, 2, 5)))
