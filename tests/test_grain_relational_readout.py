#!/usr/bin/env python3
"""Finite witnesses for information in joint KV relations rather than marginals."""

import numpy as np

from grain_research.vector_mechanisms.query_decomposition import centered_values
from grain_research.vector_mechanisms.query_tilt import query_jacobian, tilted_readout


def test_equal_marginals_can_have_opposite_relational_scores():
    k = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
    a, b = k.copy(), -k
    np.testing.assert_array_equal(np.sort(a[:, 0]), np.sort(b[:, 0]))
    np.testing.assert_array_equal(a.mean(0), b.mean(0))
    score_a = tilted_readout(k, a, np.array([0.7]))[0][0]
    score_b = tilted_readout(k, b, np.array([0.7]))[0][0]
    assert score_a > 0
    np.testing.assert_allclose(score_b, -score_a)


def test_uniform_query_annihilates_pairing_and_derivative_is_joint_moment():
    rng = np.random.default_rng(937)
    k, v = rng.normal(size=(17, 5)), rng.normal(size=(17, 4))
    groups = [[1, 2, 3, 4], [8, 9, 10, 11, 12]]
    centered = centered_values(v, groups)
    zero = np.zeros(5)
    np.testing.assert_allclose(tilted_readout(k, centered, zero)[0], 0, atol=1e-14)
    moment = np.zeros((4, 5))
    for g in groups:
        moment += (v[g] - v[g].mean(0)).T @ (k[g] - k[g].mean(0)) / len(k)
    np.testing.assert_allclose(query_jacobian(k, centered, zero), moment, atol=1e-14)
    direction, eps = rng.normal(size=5), 1e-5
    measured = (
        tilted_readout(k, centered, eps * direction)[0]
        - tilted_readout(k, centered, -eps * direction)[0]
    ) / (2 * eps)
    np.testing.assert_allclose(measured, moment @ direction, atol=1e-9)
