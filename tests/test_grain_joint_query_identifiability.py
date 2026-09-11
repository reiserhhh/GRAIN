#!/usr/bin/env python3
"""Post-NEXT88 witnesses: contrast, mean and paired functional coordinates."""

import numpy as np

from grain_research.vector_mechanisms.evidence_coordinate_reader import lift_value


def test_same_contrast_can_hide_an_equal_error_in_both_queries():
    p = np.array([[2.0, 1.0], [0.0, 1.0]])
    m, d, v = np.array([0.2, -0.1]), np.array([0.7, 0.3]), np.array([1.0, -1.0])
    scores = []
    for difference in (d, d + v):
        readers = np.stack((m + difference / 2, m - difference / 2))
        scores.append(np.einsum("qh,qh->q", p, readers))
    change = scores[1] - scores[0]
    np.testing.assert_allclose(change, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(p.mean(axis=0) @ v, 0, atol=1e-12)
    np.testing.assert_allclose(change.mean(), (p[0] - p[1]) @ v / 4)


def test_fixed_common_reader_does_not_force_absolute_query_failure():
    p = np.array([[2.0, 1.0], [0.0, 1.0]])
    m, target = np.array([0.2, -0.1]), np.array([1.7, -0.2])
    a = np.stack((p[0] / 2, -p[1] / 2))
    d = np.linalg.solve(a, target - p @ m)
    readers = np.stack((m + d / 2, m - d / 2))
    np.testing.assert_allclose(np.einsum("qh,qh->q", p, readers), target, atol=1e-12)


def test_independent_query_lifts_close_all_ten_functional_observations():
    rng = np.random.default_rng(889)
    p = rng.normal(size=(2, 5, 11))
    true, baseline, origin = rng.normal(size=(3, 2, 11))
    units = np.arange(1.0, 6.0)
    target = np.einsum("qkh,qh->qk", p, true)
    lifted = []
    for q in range(2):
        r = p[q] / units[:, None]
        c = (target[q] - p[q] @ origin[q]) / units
        lifted.append(lift_value(baseline[q], r, c, origin[q]))
    scores = np.einsum("qkh,qh->qk", p, np.stack(lifted))
    np.testing.assert_allclose(scores, target, atol=1e-10)
    np.testing.assert_allclose(scores.mean(axis=0), target.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(scores[0] - scores[1], target[0] - target[1], atol=1e-10)


def test_pair_error_separates_unobserved_mean_and_observed_contrast():
    errors = np.random.default_rng(8810).normal(size=(19, 2, 5))
    average = errors.mean(axis=1)
    contrast = errors[:, 0] - errors[:, 1]
    np.testing.assert_allclose(
        np.sum(errors**2, axis=1), 2 * average**2 + contrast**2 / 2, atol=1e-12
    )
