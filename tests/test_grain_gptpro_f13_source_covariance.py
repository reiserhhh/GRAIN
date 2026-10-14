#!/usr/bin/env python3
"""Source-only covariance fitting and identifiable design checks."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_source_covariance import (
    covariance_basis,
    fit_source_covariance,
)


def test_exact_covariance_components():
    """Construct centered source observations with exact desired sample covariance."""
    x = np.column_stack([np.ones(8), np.tile([-1, -1 / 3, 1 / 3, 1], 2)])
    c = 0.6 ** abs(np.arange(8)[:, None] - np.arange(8))
    v = np.array([[1.0, 0.1], [0.1, 0.25]])
    k = x @ v @ x.T + 0.25 * c
    q = np.linalg.qr(
        np.column_stack([np.ones(10), np.random.default_rng(12).normal(size=(10, 8))])
    )[0][:, 1:]
    values = np.sqrt(9) * q @ np.linalg.cholesky(k).T + x @ np.array([0.2, 0.5])
    fit = fit_source_covariance(values, x, c)
    np.testing.assert_allclose(fit["prior"], v, atol=1e-12)
    np.testing.assert_allclose(fit["sigma2"], 0.25, atol=1e-12)
    np.testing.assert_allclose(fit["mean"], [0.2, 0.5], atol=1e-12)
    assert fit["status"] == "SUPPORTED"


def test_unidentifiable_design_rejected():
    """One fixed condition cannot identify level and response covariance."""
    with pytest.raises(ValueError, match="identify"):
        covariance_basis(np.ones((8, 2)), np.eye(8))
