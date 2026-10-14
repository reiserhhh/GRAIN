#!/usr/bin/env python3
"""Falsifiable controls for observation equivalence and author aggregation."""

import numpy as np
import pytest

from grain_research.construction import math_author_evidence_merge as M


def test_any_query_local_key_translation_invariance():
    """Arbitrary shared key translation changes partitions, not local readouts."""
    rng = np.random.default_rng(102)
    k, v = rng.normal(size=(5, 2, 4)), rng.normal(size=(5, 2, 4))
    q, d = rng.normal(size=(3, 4, 4)), rng.normal(size=(3, 4, 4))
    delta = rng.normal(size=(2, 4))
    f, z = M.observe(k, v, q, d)
    fg, zg = M.observe(k + delta, v, q, d)
    np.testing.assert_allclose(fg, f, atol=1e-13)
    np.testing.assert_allclose(
        zg - z, np.einsum("thd,hd->th", q, np.repeat(delta, 2, 0))
    )


def test_common_battery_offset_is_query_only():
    """A realizable vector witness simultaneously shifts every registered head."""
    q = np.eye(6).reshape(3, 2, 6)
    shift, diag = M.common_key_offset(q, 1)
    np.testing.assert_allclose(np.einsum("thd,hd->th", q, np.repeat(shift, 2, 0)), 1)
    assert diag[0]["residual"] < 1e-14


def test_infeasible_common_offset_is_reported_not_repaired():
    """Opposite queries cannot both receive a unit key-translation shift."""
    _, diag = M.common_key_offset(np.array([[[1.0]], [[-1.0]]]), 1)
    assert diag[0]["residual"] > 0.9


def test_covariance_gap_and_distinct_estimands():
    """A partition-weighted author law is not the declared author composition."""
    f, z, p = np.array([0.0, 2.0]), np.log([1.0, 3.0]), np.array([0.5, 0.5])
    expected, merged, cov = M.compose(f, z, p)
    assert expected == 1
    assert merged == 1.5
    np.testing.assert_allclose(merged - expected, cov)
    np.testing.assert_allclose(M.compose(f, z + 1000, p)[1], merged)


def test_local_scores_do_not_determine_cross_evidence_merge():
    """Equivalent local functions can give distinct author merges."""
    f, p = np.array([0.0, 1.0]), np.array([0.5, 0.5])
    first = M.compose(f, np.zeros(2), p)
    second = M.compose(f, np.array([0.0, np.log(3)]), p)
    assert first[0] == second[0] == 0.5
    assert first[1] == 0.5 and second[1] == 0.75


def test_no_history_null_can_be_manufactured_by_partition_merge():
    """Matched current marginals imply no declared effect, unlike arbitrary lifts."""
    p, m = M.author_laws()
    f = np.broadcast_to(np.arange(3), (3, 3)).ravel()
    z = np.repeat([-0.5, 0, 0.5], 3)
    plus, minus = M.compose(f, z, p), M.compose(f, z, m)
    np.testing.assert_allclose(plus[0] - minus[0], 0, atol=1e-15)
    assert abs(plus[1] - minus[1]) > 0.01


def test_positive_lift_associative_reorder_and_extreme_values():
    """Stable lift addition preserves a declared common scale and split boundaries."""
    a, b, c = (1000.0, -3.0), (999.0, 7.0), (-1000.0, 2.0)
    left = M.merge_lifts(M.merge_lifts(a, b), c)
    right = M.merge_lifts(a, M.merge_lifts(b, c))
    reverse = M.merge_lifts(c, M.merge_lifts(b, a))
    np.testing.assert_allclose(left, right, atol=1e-12)
    np.testing.assert_allclose(left, reverse, atol=1e-12)


def test_token_duplication_and_unit_mean_are_distinct_contracts():
    """Duplication preserves local reading but changes token-sum evidence mass."""
    k, v, q, d = (
        np.ones((2, 1, 1)),
        np.ones((2, 1, 1)),
        np.ones((1, 1, 1)),
        np.ones((1, 1, 1)),
    )
    f, z = M.observe(k, v, q, d)
    f2, z2 = M.observe(np.repeat(k, 2, 0), np.repeat(v, 2, 0), q, d)
    np.testing.assert_allclose(f2, f)
    np.testing.assert_allclose(z2 - z, np.log(2))
    np.testing.assert_allclose(z2 - np.log(4), z - np.log(2))


def test_fail_closed_empty_nonfinite_and_invalid_law():
    """Missing evidence is not silently fabricated as zero or uniform evidence."""
    with pytest.raises(ValueError):
        M.observe(
            np.zeros((0, 1, 1)),
            np.zeros((0, 1, 1)),
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
        )
    with pytest.raises(ValueError):
        M.compose(np.ones(2), np.zeros(2), [0.5, -0.5])
    with pytest.raises(ValueError):
        M.compose([np.nan, 1], [0, 0], [0.5, 0.5])


def test_cold_boundaries_checked():
    """A numeric export cannot hide an empty or overlapping record boundary."""
    with pytest.raises(ValueError):
        M.cold_score(
            {},
            {
                "keys": np.zeros((2, 1, 1)),
                "values": np.zeros((2, 1, 1)),
                "offsets": [0, 0, 2],
                "endpoints": [1, 1],
            },
        )
