#!/usr/bin/env python3
"""Source moment and Gaussian conditional scoring algebra checks."""

import numpy as np

from scripts.run_grain_gptpro_f13_delta_nested import fit_moments, fit_reml, posterior


def test_precision_posterior_matches_observation_covariance():
    rng = np.random.default_rng(13)
    x = np.column_stack([np.ones(24), np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 6)])
    mu = np.array([0.1, 0.5])
    covariance = np.array([[1.0, 0.1], [0.1, 0.25]])
    y = rng.normal(size=(10, 24))
    m, v = posterior(mu, covariance, 0.25, y, x)
    gain = covariance @ x.T @ np.linalg.inv(x @ covariance @ x.T + 0.25 * np.eye(24))
    np.testing.assert_allclose(m, mu + (y - mu @ x.T) @ gain.T, atol=1e-12)
    np.testing.assert_allclose(v, covariance - gain @ x @ covariance, atol=1e-12)


def test_source_moments_match_balanced_scalar_ols():
    rng = np.random.default_rng(14)
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 8)
    x = np.column_stack([np.ones(32), z])
    y = rng.normal(size=(300, 32))
    mu, cov, noise, _eigen = fit_moments(y, x)
    coefficients = np.column_stack([y.mean(axis=1), y @ z / (z @ z)])
    np.testing.assert_allclose(mu, coefficients.mean(axis=0), atol=1e-12)
    expected = np.sum((y - coefficients @ x.T) ** 2) / (300 * 30)
    np.testing.assert_allclose(noise, expected, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() >= 1e-6 - 1e-12


def test_bootstrap_copies_are_distinct_random_effect_groups(monkeypatch):
    """Repeated sampled authors must not collapse into one mixed-model group."""
    from grain_research.gptpro_foundation_v3 import f04_recovery

    captured = []

    def fake_fit(rows):
        captured.extend(rows)
        return {"mu": [0, 0], "Sigma": [[1, 0], [0, 1]], "sigma2": 1}

    monkeypatch.setattr(f04_recovery, "fit_source_lmm", fake_fit)
    x = np.column_stack([np.ones(4), [-1, -0.3, 0.3, 1]])
    fit_reml(np.ones((3, 4)), x)
    assert len({r["author_id"] for r in captured}) == 3
    assert len({r["event_id"] for r in captured}) == 12
    assert {r["role"] for r in captured} == {"source"}


def test_balanced_interior_reml_agrees_with_exact_moments():
    """Independent statsmodels optimizer approaches the balanced closed form."""
    rng = np.random.default_rng(155)
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 8)
    x = np.column_stack([np.ones(32), z])
    beta = rng.normal(size=(40, 2))
    y = beta @ x.T + rng.normal(scale=0.5, size=(40, 32))
    expected = fit_moments(y, x)
    actual = fit_reml(y, x)
    assert not actual[4]["fallback"]
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-8)
    np.testing.assert_allclose(actual[1], expected[1], atol=0.002)
    np.testing.assert_allclose(actual[2], expected[2], atol=0.0001)
