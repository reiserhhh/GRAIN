#!/usr/bin/env python3
"""Synthetic counterexamples for typed author and evidence-action correspondence."""

import numpy as np
import pytest

from grain_research.construction import math_event_behavior as M


def test_positive_author_association_does_not_imply_same_event_action():
    """An unobserved next event can set author position opposite to the current change."""
    r = np.repeat(np.array([-3, -1, 1, 3])[:, None], 5, axis=1)
    assert np.diag(M.contacts(r, r)["covariance"]).tolist() == [5] * 5
    assert np.diag(M.contacts(-r, r)["action"]).tolist() == [-5] * 5


def test_reversal_and_components_remain():
    """Reversing both endpoints preserves actions, not an unchanged author association."""
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(2, 12, 5))
    a, b = M.contacts(x, y), M.contacts(-x, -y)
    for key in a:
        np.testing.assert_allclose(a[key], b[key])
    np.testing.assert_allclose(a["action"], a["covariance"] + a["means"], atol=1e-15)
    np.testing.assert_allclose(M.contacts(x, -y)["covariance"], -a["covariance"])


def test_finite_support_retains_zero_multiplicity():
    """Zero field change can be qualified without choosing an arbitrary cell identity."""
    rng = np.random.default_rng(8)
    w, t = rng.normal(size=(2, 8, 5))
    rows = np.array([w[2] - w[6], np.zeros(5)])
    got, counts = M.support_transfer(rows, w, t, "difference")
    np.testing.assert_allclose(got, [t[2] - t[6], np.zeros(5)])
    assert counts == [1, 8]
    shifted, _ = M.support_transfer(rows, w + 4, t - 9, "difference")
    np.testing.assert_allclose(got, shifted)


def test_nonidentified_support_is_not_averaged():
    """Collapsing source cells cannot identify an action on an uncollapsed other law."""
    w = np.zeros((8, 5))
    t = np.arange(40).reshape(8, 5)
    with pytest.raises(ValueError, match="not identified"):
        M.support_transfer(np.zeros((1, 5)), w, t, "difference")
    with pytest.raises(ValueError, match="No matching"):
        M.support_transfer(np.ones((1, 5)), w, t, "difference")


def test_means_do_not_require_choosing_an_order():
    """Both pair orders imply the identical transported author mean."""
    rng = np.random.default_rng(9)
    w, t = rng.normal(size=(2, 8, 5))
    got, counts = M.support_transfer([(w[2] + w[6]) / 2], w, t, "mean")
    np.testing.assert_allclose(got, [(t[2] + t[6]) / 2])
    assert counts == [2]


def test_paired_bootstrap_resamples_authors_not_directions():
    """Exactly opposite products in dependent directions cancel for every resample."""
    x = np.ones((2, 12, 5))
    y = x.copy()
    y[1] = -1
    np.testing.assert_array_equal(M.paired_intervals(x, y, 10, 99), np.zeros((5, 2)))
