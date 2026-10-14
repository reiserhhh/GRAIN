#!/usr/bin/env python3
"""F05 latent-state checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import dense_loglik, eta_from_rho, generate_world, kalman_filter, rho_from_eta


def test_rho_transform_roundtrip_and_stationarity():
    for rho in (.2, .5, .8, .95):
        np.testing.assert_allclose(rho_from_eta(eta_from_rho(rho)), rho, atol=1e-12)
        assert abs(rho_from_eta(rho)) < 100


def test_filter_matches_dense_likelihood():
    rng = np.random.default_rng(2)
    z = np.where(np.arange(12) % 2, 1.0, -1.0)
    y = rng.normal(size=12)
    k = kalman_filter(y, z, a=.1, b=.3, rho=.6, sigma_w=.5, sigma_e=.5)
    d = dense_loglik(y, z, a=.1, b=.3, rho=.6, sigma_w=.5, sigma_e=.5)
    np.testing.assert_allclose(k["loglik"], d, atol=1e-8)


def test_world_has_truth_separate_from_rows():
    rows, truth = generate_world("near_root", 7, counts={"source": 1, "validation": 1, "application": 1}, events=8)
    assert len(rows) == 24
    assert len(truth) == 3
    assert all("rho" not in row for row in rows)
