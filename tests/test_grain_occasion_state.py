"""Tests for the GRAIN-C4-02 occasion-state Kalman filter."""

import numpy as np
from scipy.stats import multivariate_normal

from grain_research.construction import occasion_state as OS


def _cov(t, lam, q, s2):
    return q * np.exp(-lam * np.abs(t[:, None] - t[None, :])) + s2 * np.eye(len(t))


def test_kalman_loglik_equals_dense_gaussian():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(5.0, 12))
    lam, q, s2 = 0.1, 0.5, 0.8
    r = rng.multivariate_normal(np.zeros(12), _cov(t, lam, q, s2))
    ll = OS.loglik((lam, q, s2), r[None], np.diff(t)[None])
    assert abs(ll - multivariate_normal(np.zeros(12), _cov(t, lam, q, s2)).logpdf(r)) < 1e-9


def test_prequential_mean_equals_dense_conditional():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.exponential(3.0, 8))
    p = {"lambda_per_day": 0.2, "q": 0.6, "sigma2": 0.4}
    c = _cov(t, 0.2, 0.6, 0.4)
    r = rng.multivariate_normal(np.zeros(8), c)
    m, v = OS.prequential(p, r, np.diff(t), 5)
    for k, s in enumerate(range(5, 8)):
        w = np.linalg.solve(c[:s, :s], c[:s, s])
        assert abs(m[k] - w @ r[:s]) < 1e-9
        assert abs(v[k] - (c[s, s] - c[:s, s] @ w)) < 1e-9


def test_fit_recovers_ou_parameters():
    rng = np.random.default_rng(2)
    lam, q, s2 = 0.05, 0.6, 0.4
    R, G = [], []
    for _ in range(400):
        t = np.cumsum(rng.exponential(8.0, 25))
        R.append(rng.multivariate_normal(np.zeros(25), _cov(t, lam, q, s2)))
        G.append(np.diff(t))
    f = OS.fit(np.array(R), np.array(G))
    assert abs(f["q"] - q) < 0.12 and abs(f["sigma2"] - s2) < 0.1 and 0.02 < f["lambda_per_day"] < 0.12
