#!/usr/bin/env python3
"""F11-03 variance-0 boundary: REML at true method variance 0."""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from grain_research.gptpro_foundation_v3.f11_mtmm import (
    designs,
    fit_mtmm,
    mtmm_covariance,
)

SIGMA = np.full((3, 3), 0.3) + 0.7 * np.eye(3)


def reml_objective(y, sigma, method, error, event=0.0):
    """Same REML objective as fit_mtmm (unknown 18-cell mean)."""
    centered = y - y.mean(axis=0)
    v = mtmm_covariance(sigma, method, event, error)
    chol = cho_factor(v, lower=True)
    return 0.5 * ((len(y) - 1) * 2 * np.log(np.diag(chol[0])).sum() + np.trace(cho_solve(chol, centered.T @ centered)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_zero_method_variance_is_reported_at_boundary_not_negative(seed):
    rng = np.random.default_rng(seed)
    f, _, _ = designs(False)
    trait = rng.multivariate_normal(np.zeros(3), SIGMA, 300)
    y = trait @ f.T + rng.normal(scale=0.5, size=(300, 18))
    fit = fit_mtmm(y, conditional=False, include_event=False)
    assert fit["success"]
    assert 0 <= fit["method"] <= 0.01
    assert abs(fit["error"] - 0.25) < 0.02
    base = reml_objective(y, fit["sigma"], fit["method"], fit["error"])
    for alternative in (1e-3, 1e-2, 5e-2):
        assert reml_objective(y, fit["sigma"], alternative, fit["error"]) >= base - 1e-9


def test_positive_method_variance_is_recovered_away_from_boundary():
    rng = np.random.default_rng(7)
    f, m, _ = designs(False)
    trait = rng.multivariate_normal(np.zeros(3), SIGMA, 300)
    method = rng.normal(scale=0.5, size=(300, 3))
    y = trait @ f.T + method @ m.T + rng.normal(scale=0.5, size=(300, 18))
    fit = fit_mtmm(y, conditional=False, include_event=False)
    assert fit["success"] and 0.15 < fit["method"] < 0.35
