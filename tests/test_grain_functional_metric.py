#!/usr/bin/env python3
"""Exact algebra tests for response geometry, with singular support allowed."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_metric import functional_metric


def fixture():
    """A fixed full-rank coordinate map and finite multiquery reference support."""
    rng = np.random.default_rng(22)
    return rng, rng.normal(size=(4, 9)), rng.normal(size=(12, 4, 9))


def test_gram_psd_exact_split_and_reference_orthogonality():
    """Both components survive while their declared reference covariance vanishes."""
    _rng, g, d = fixture()
    result = functional_metric(g, d, 3)
    assert np.min(result["eigenvalues"]) >= -1e-12
    np.testing.assert_allclose(result["shared"] + result["specific"], g, atol=1e-14)
    np.testing.assert_allclose(result["reference_anchor_covariance"], 0, atol=1e-14)
    np.testing.assert_allclose(result["specific"][3], 0, atol=1e-14)
    np.testing.assert_allclose(np.diag(result["normalized"]), 1)


def test_covector_coordinate_change_preserves_function_metric():
    """A nonorthogonal basis change alters coordinates but not the readout geometry."""
    rng, g, d = fixture()
    q, _r = np.linalg.qr(rng.normal(size=(9, 9)))
    transform = q @ np.diag(np.linspace(0.2, 4, 9))
    mapped_d = d @ transform.T
    mapped_g = np.linalg.solve(transform.T, g.T).T
    original = functional_metric(g, d, 3)
    mapped = functional_metric(mapped_g, mapped_d, 3)
    for key in ("gram", "normalized", "coefficients", "reference_scores"):
        np.testing.assert_allclose(original[key], mapped[key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        mapped["specific"],
        np.linalg.solve(transform.T, original["specific"].T).T,
        rtol=1e-12,
        atol=1e-12,
    )


def test_unexcited_reference_directions_and_zero_anchor():
    """Empirical equivalence does not require inverting a singular covariance."""
    _rng, g, d = fixture()
    d[..., 4:] = 0
    other = g.copy()
    other[:, 4:] += 100
    np.testing.assert_allclose(
        functional_metric(g, d, 3)["gram"],
        functional_metric(other, d, 3)["gram"],
    )
    with pytest.raises(ValueError, match="zero reference"):
        functional_metric(g, np.zeros_like(d), 3)
