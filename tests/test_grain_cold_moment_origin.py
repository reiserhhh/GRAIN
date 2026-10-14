#!/usr/bin/env python3
"""Origin covariance is a derived account, not deletion of the author mean."""

import numpy as np

from scripts.diagnose_grain_cold_moment_origin import covariance_pair


def test_origin_translation_preserves_covariance_and_keeps_mean():
    rng = np.random.default_rng(71004)
    p = rng.dirichlet(np.ones(8), size=2)
    z = rng.normal(size=(8, 5))
    b = np.array([2.0, -3.0, 4.0, 5.0, -6.0])
    first, c0, error0 = covariance_pair(p, z)
    second, c1, error1 = covariance_pair(p, z + b)
    assert max(error0, error1) < 1e-12
    assert np.allclose(c0, c1)
    assert np.allclose(second["means"], first["means"] + b)
    delta = first["means"][1] - first["means"][0]
    changed = (
        second["second"][1]
        - second["second"][0]
        - first["second"][1]
        + first["second"][0]
    )
    assert np.allclose(changed, np.outer(b, delta) + np.outer(delta, b))


def test_same_mean_different_dependence_survives_origin_account():
    p = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]])
    z = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    means, covariance, error = covariance_pair(p, z)
    assert error == 0
    assert np.allclose(means["means"], 0)
    assert np.linalg.norm(covariance[1] - covariance[0]) > 0
