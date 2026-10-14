#!/usr/bin/env python3
"""Focused conditional criterion tests, including an exhaustive finite orbit."""

from itertools import permutations, product

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as M
from grain_research.construction.math_source_event_coupling import BITS
from scripts.score_grain_criterion_components_cold import score


def fixture():
    """All32 profile sides are supported in both synthetic SOURCE banks."""
    labels = np.tile(BITS, (2, 1))
    banks = np.repeat((0, 1), 32)
    rng = np.random.default_rng(20)
    d = rng.normal(size=(2, 64, 3, 64))
    d.transpose(0, 2, 1, 3)[:, :, banks[:, None] == banks] = 0
    return labels, banks, d


def test_decomposition_and_criterion_centering():
    y, b, d = fixture()
    values, means, e, groups = M.score_components(d, y, b)
    np.testing.assert_allclose(
        values[..., 0, :], values[..., 1, :] + values[..., 2, :], atol=1e-13
    )
    np.testing.assert_allclose(means, 0.5)
    np.testing.assert_allclose(M.group_center(e, groups), e)


def test_application_criterion_does_not_enter_own_score():
    y, b, d = fixture()
    old = M.score_components(d, y, b)[0]
    changed = y.copy()
    changed[b == 0] = 1 - changed[b == 0]
    new = M.score_components(d, changed, b)[0]
    np.testing.assert_array_equal(old[:, b == 0], new[:, b == 0])
    assert np.max(abs(old[:, b == 1] - new[:, b == 1])) > 0


def test_own_bank_and_missing_support_rejected():
    y, b, d = fixture()
    d[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="cross-bank"):
        M.score_components(d, y, b)
    with pytest.raises(ValueError, match="Unobserved"):
        M.criterion_design(y[1:], b[1:])


def test_exhaustive_bilinear_orbit_variance():
    y = np.tile([[0], [1], [0], [1]], (1, 5)).astype(float)
    b = np.array([0, 0, 1, 1])
    g = np.zeros_like(y, int)
    d = np.array([[0, 0, 2, 3], [0, 0, 1, -2], [4, -1, 0, 0], [3, 7, 0, 0]], float)
    e = y[:, 0] - 0.5
    values = []
    for a, bb in product(permutations(range(2)), repeat=2):
        ix = np.r_[list(a), np.array(bb) + 2]
        v = e[ix]
        values.append(v @ d @ v / 4)
    assert abs(np.mean(values)) < 1e-15
    np.testing.assert_allclose(
        M.orbit_variance(d, y, b, 0, g), np.var(values), atol=1e-15
    )


def test_assignment_preserves_strata_and_profile_counts():
    y, b, d = fixture()
    observed, null, var, p, holm, assignments = M.assignment_reference(
        d[:, :, 2], y, b, 7, 28
    )
    assert observed.shape == var.shape == p.shape == holm.shape == (2, 5)
    assert null.shape == (7, 2, 5)
    for q in range(5):
        for ix in assignments[q]:
            np.testing.assert_array_equal(np.sort(ix), np.arange(64))
            np.testing.assert_array_equal(b[ix], b)
            np.testing.assert_array_equal(
                y[ix][:, [k for k in range(5) if k != q]],
                y[:, [k for k in range(5) if k != q]],
            )
    assert np.all(holm >= p)


def test_constant_conditional_score_is_undefined_not_zero():
    y, b, _ = fixture()
    _, e, g = M.criterion_design(y, b)
    assert np.isnan(M.conditional_r(np.ones_like(y), e, g)).all()


def test_cold_rejects_application_labels_before_loading():
    with pytest.raises(ValueError, match="no evaluated labels"):
        score(
            "missing.npz",
            {
                "author_cache_id": 0,
                "sampling_regime": "next_cached_record_within_registered_window",
                "pairs": [],
                "label": 1,
            },
            1,
            "GRAIN15",
        )
