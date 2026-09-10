#!/usr/bin/env python3
"""Known-score path integral and ordered product-error identity."""

import numpy as np

from grain_research.vector_mechanisms.query_pair import scores
from scripts.analyze_grain_query_pair_direction import (
    direction_account,
    score_gradient,
    score_path,
)


def test_output_gradient_matches_central_difference():
    rng = np.random.default_rng(91)
    z, d = rng.normal(size=(8, 10)), rng.normal(size=(8, 10))
    v = np.linspace(0, 1, 10)
    eps = 1e-5
    fd = (scores(z + eps * d, v) - scores(z - eps * d, v)) / (2 * eps)
    np.testing.assert_allclose(fd, np.sum(score_gradient(z, v) * d, 1), atol=1e-10)


def test_integrated_score_gradient_recovers_nonlinear_action():
    rng = np.random.default_rng(92)
    z, d = rng.normal(size=(8, 10)), rng.normal(size=(8, 10)) * 0.8
    result = score_path(z, d, np.linspace(0, 1, 10))
    assert result["identity_max"] < 1e-12


def test_identical_norm_opposite_direction_has_orientation_error():
    rng = np.random.default_rng(93)
    z, d = rng.normal(size=(8, 10)), rng.normal(size=(8, 10)) * 0.1
    t, p, e, identity = direction_account(z, d, -d, np.linspace(0, 1, 10))
    np.testing.assert_allclose(e["magnitude"], 0, atol=1e-15)
    assert np.linalg.norm(e["orientation"]) > 0
    np.testing.assert_allclose(sum(e.values()), p["action"] - t["action"], atol=1e-12)
    assert identity["ordered_account_max"] < 1e-12


def test_zero_displacement_is_well_defined():
    z = np.zeros((5, 4))
    d = np.tile(np.arange(4) / 10, (5, 1))
    for a, pred in ((z, d), (d, z), (z, z)):
        t, p, e, identity = direction_account(z, a, pred, np.linspace(0, 1, 4))
        assert identity["ordered_account_max"] < 1e-12
        assert all(np.isfinite(v).all() for v in e.values())
        np.testing.assert_allclose(
            sum(e.values()), p["action"] - t["action"], atol=1e-12
        )
