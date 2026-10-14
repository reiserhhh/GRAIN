"""Exact covariance, identifiability and source-only scoring checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f11_mtmm import (
    designs,
    moment_fit,
    mtmm_covariance,
    predict_traits,
)


def test_actual_crossed_components():
    sigma = np.full((3, 3), 0.3)
    np.fill_diagonal(sigma, 1.0)
    v = mtmm_covariance(sigma, event=0.25)
    assert v[0, 0] == 1.75
    assert v[0, 1] == 1.25  # same trait/reader, different occasion
    assert v[0, 2] == 1.25  # same trait/occasion, different reader
    assert v[0, 9] == 0.3  # different trait/reader/occasion


def test_constant_shared_method_is_nonidentified():
    s = np.eye(3)
    np.testing.assert_allclose(
        mtmm_covariance(s, actor=0.25), mtmm_covariance(s + 0.25 * np.ones((3, 3)))
    )


def test_independent_anchor_breaks_constant_method_equivalence():
    f, _, _ = designs()
    s = np.eye(3)
    # Cov(y, independent noisy trait anchor) differs despite identical Cov(y).
    assert np.max(np.abs(f @ s - f @ (s + 0.25 * np.ones((3, 3))))) == 0.25


def test_blup_dense_conditional_identity():
    rng = np.random.default_rng(2)
    model = {
        "sigma": np.eye(3),
        "method": 0.25,
        "event": 0.25,
        "error": 0.25,
        "mean": np.zeros(18),
    }
    y = rng.normal(size=(10, 18))
    f, _, _ = designs(True)
    v = mtmm_covariance(np.eye(3), event=0.25, conditional=True)
    p, uncertainty = predict_traits(model, y, True)
    np.testing.assert_allclose(p, y @ np.linalg.inv(v) @ f, atol=1e-12)
    np.testing.assert_allclose(
        uncertainty, np.eye(3) - f.T @ np.linalg.inv(v) @ f, atol=1e-12
    )


def test_moment_basis_full_rank():
    result = moment_fit(np.random.default_rng(2).normal(size=(300, 18)))
    assert result["design_rank"] == 9
    assert np.linalg.eigvalsh(result["sigma"]).min() > 0
