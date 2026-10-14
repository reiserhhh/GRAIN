"""Independent algebra and leakage checks for F04 residual inference."""

import numpy as np
import pytest
from scipy.stats import chi2

from grain_research.gptpro_foundation_v3.f04_tail_recovery import predict_tail


def sample():
    """Return a reproducible observed-only linear prefix."""
    z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 6)
    y = 0.2 + 0.7 * z + np.random.default_rng(4).normal(0, 0.5, len(z))
    return [
        {"z": float(a), "y": float(b), "event_id": str(i)}
        for i, (a, b) in enumerate(zip(z, y))
    ]


def test_gaussian_against_direct_lstsq():
    rows = sample()
    result = predict_tail(rows)
    x = np.array([[1, row["z"]] for row in rows])
    y = np.array([row["y"] for row in rows])
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    rss = residual @ residual
    assert result["variance"] == pytest.approx(rss / 22)
    assert result["variance_ci_low"] == pytest.approx(rss / chi2.ppf(0.975, 22))
    assert result["variance_ci_high"] == pytest.approx(rss / chi2.ppf(0.025, 22))


def test_bootstrap_against_explicit_refits():
    rows = sample()
    result = predict_tail(rows, interval="bootstrap", draws=80, seed=9)
    x = np.array([[1, row["z"]] for row in rows])
    y = np.array([row["y"] for row in rows])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    h = x @ np.linalg.inv(x.T @ x) @ x.T
    pool = residual / np.sqrt(1 - np.diag(h))
    pool -= pool.mean()
    samples = np.random.default_rng(9).choice(pool, (80, 24))
    estimates = []
    for noise in samples:
        response = x @ beta + noise
        e = response - x @ np.linalg.lstsq(x, response, rcond=None)[0]
        estimates.append(e @ e / 22)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    assert result["variance_ci_low"] == pytest.approx(
        max(0, 2 * result["variance"] - hi)
    )
    assert result["variance_ci_high"] == pytest.approx(
        max(0, 2 * result["variance"] - lo)
    )


def test_rank_and_invalid_input():
    rows = [{**row, "z": -1.0} for row in sample()]
    assert predict_tail(rows)["df"] == 23
    with pytest.raises(ValueError, match="degrees"):
        predict_tail(sample()[:2])
    with pytest.raises(ValueError):
        predict_tail(sample(), interval="oracle")


def test_latent_fields_unused():
    rows = sample()
    changed = [{**row, "sigma2": 999, "truth": -999} for row in rows]
    assert predict_tail(rows) == predict_tail(changed)
