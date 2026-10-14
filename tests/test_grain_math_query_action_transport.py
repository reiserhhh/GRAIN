"""Synthetic tests for HM22 source anchors and signed action accounts."""

import numpy as np
import pytest

from grain_research.construction import math_query_action_transport as Q


def test_source_anchor_is_family_balanced_and_rejects_held():
    """Duplicate source opportunities retain their declared measure, not independence."""
    rows = [{"role": "source", "query": "q", "family": f} for f in (0, 0, 1)]
    g = np.array([[1, -1], [3, -3], [6, -6]])
    a, w = Q.source_anchor(rows, g)
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5])
    np.testing.assert_allclose(a, [4, -4])
    with pytest.raises(ValueError, match="held"):
        Q.source_anchor([{**rows[0], "role": "application"}], g[:1])
    with pytest.raises(ValueError, match="one query"):
        Q.source_anchor([rows[0], {**rows[1], "query": "other"}], g[:2])


def test_constant_reference_sensitivity():
    """Reference-invariant score covectors give G=0 without fitting any target."""
    d, a = np.array([[1, -1], [2, -2], [-3, 3]]), np.array([0.2, -0.2])
    p = Q.accounts(d, np.tile(a, (3, 1)), a, np.zeros(3))
    np.testing.assert_allclose(p[:, 0], d @ a)
    np.testing.assert_array_equal(p[:, 1:], 0)


def test_reference_variation_can_reverse_covariance():
    """Identical displacement covariance need not imply identical score covariance."""
    d = np.array([[-1, 1], [0, 0], [1, -1]])
    anchor = np.array([0.2, -0.2])
    left = Q.accounts(d, np.tile(anchor, (3, 1)), anchor, np.zeros(3))
    right = Q.accounts(d, np.tile(-anchor, (3, 1)), anchor, np.zeros(3))
    v = Q.covariance_accounts(left, right, np.full(3, 1 / 3))
    assert v["fixed_covariance"] > 0 > v["native_covariance"]
    assert not v["fixed_native_sign_agree"]
    assert v["recombination_error"] < 1e-15


def test_gauge_alphabet_units_and_noop():
    """Fixed zero-sum covectors annihilate common logit shifts and transform units."""
    d = np.array([[0.1, -0.3, 0.2], [0, 0, 0]])
    g, a = np.array([[0.1, 0.2, -0.3], [0.2, 0.1, -0.3]]), np.array([0.3, -0.2, -0.1])
    c = np.array([0.05, 0.0])
    p = Q.accounts(d, g, a, c)
    np.testing.assert_allclose(Q.accounts(d + 5, g, a, c), p, atol=1e-15)
    np.testing.assert_allclose(Q.accounts(d, 3 * g, 3 * a, 3 * c), 3 * p, atol=1e-15)
    np.testing.assert_array_equal(p[1], 0)


def test_nine_terms_and_displacement_matrix():
    """Scalar contractions reproduce the complete vector covariance and all cross terms."""
    rng = np.random.default_rng(22)
    da, db = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
    aa, ab = rng.normal(size=4), rng.normal(size=4)
    ga, gb = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
    pa = Q.accounts(da, ga, aa, rng.normal(size=12))
    pb = Q.accounts(db, gb, ab, rng.normal(size=12))
    w = np.full(12, 1 / 12)
    v = Q.covariance_accounts(pa, pb, w)
    assert v["recombination_error"] < 1e-13
    assert v["BB"] == pytest.approx(
        aa @ Q.displacement_cross_covariance(da, db, w) @ ab, abs=1e-13
    )
    assert v["native_covariance"] == pytest.approx(
        np.cov(pa.sum(axis=1), pb.sum(axis=1), bias=True)[0, 1], abs=1e-13
    )


def test_zero_native_with_nonzero_cancellation():
    """Unknown contribution ratios stay undefined when nonzero terms cancel exactly."""
    v = np.array([[1, -1, 0], [2, -2, 0], [-1, 1, 0]], float)
    result = Q.covariance_accounts(v, v, np.full(3, 1 / 3))
    assert result["BB"] > 0
    assert result["native_covariance"] == 0
    assert result["link_native_ratio"] is None
    assert result["sensitivity_native_ratio"] is None


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "curvature", "empty"])
def test_invalid_inputs(kind):
    """Reject nonfinite/incompatible arrays rather than emitting a fake zero score."""
    d, g, a, c = np.ones((2, 3)), np.ones((2, 3)), np.ones(3), np.ones(2)
    if kind == "shape":
        g = g[:, :2]
    elif kind == "nonfinite":
        d[0, 0] = np.nan
    elif kind == "curvature":
        c = c[:1]
    else:
        with pytest.raises(ValueError):
            Q.source_anchor([], np.empty((0, 3)))
        return
    with pytest.raises(ValueError):
        Q.accounts(d, g, a, c)
