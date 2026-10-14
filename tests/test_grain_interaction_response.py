#!/usr/bin/env python3
"""Instrument identification, observational equivalence and text-act scope tests."""

import numpy as np
import pytest

from grain_research.interaction_response import (
    apply_response,
    designs,
    estimate_response,
    hac_meat,
    opportunity_profile,
)


def sample(n=2000):
    """Generated common-cause fixture; truth is not an estimator argument."""
    rng = np.random.default_rng(16)
    r, z = rng.choice([-1.0, 1.0], (2, n))
    s = rng.normal(0, 0.5, n)
    u = r + 0.9 * s + rng.normal(0, 0.3, n)
    truth = np.array([0.1, 0.2, 0.3, 0.1])
    x, _ = designs(z, u, r)
    y = x @ truth + s + rng.normal(0, 0.1, n)
    return y, z, u, r, truth


def test_iv_and_control_function_recover_same_coefficients():
    y, z, u, r, truth = sample()
    fit = estimate_response(y, z, u, r)
    np.testing.assert_allclose(fit["iv"], fit["cf"][:4], atol=1e-10)
    assert np.linalg.norm(fit["iv"] - truth) < 0.05
    assert abs(fit["ols"][2] - truth[2]) > 0.1
    assert np.linalg.eigvalsh(fit["iv_cov"]).min() >= -1e-12


def test_direct_assignment_has_same_observational_alternative():
    """When u=r+v, direct r is indistinguishable from changed u plus residual loading."""
    rng = np.random.default_rng(16)
    r, v = rng.normal(size=(2, 100))
    u, c, delta, kappa = r + v, 0.3, 0.4, 0.5
    left = c * u + delta * r + kappa * v
    right = (c + delta) * u + (kappa - delta) * v
    np.testing.assert_allclose(left, right, atol=1e-12)


def test_no_assignment_and_constant_input_are_not_causal_scores():
    y, z, u, r, _ = sample(64)
    fit = estimate_response(y, z, u)
    assert fit["status"] == "observational_only" and "iv" not in fit
    fixed = estimate_response(y, z, np.ones(len(u)), r)
    assert fixed["status"] == "conditional_sum_only" and "ols" not in fixed


def test_independent_two_stage_equations():
    y, z, u, r, _ = sample(200)
    x, w = designs(z, u, r)
    projection = w @ np.linalg.inv(w.T @ w) @ w.T
    beta = np.linalg.solve(x.T @ projection @ x, x.T @ projection @ y)
    fit = estimate_response(y, z, u, r, lags=0)
    np.testing.assert_allclose(fit["iv"], beta, atol=1e-11)
    bread = np.linalg.inv(w.T @ x)
    e = y - x @ beta
    expected = bread @ w.T @ np.diag(e**2) @ w @ bread.T
    np.testing.assert_allclose(fit["iv_cov"], expected, atol=1e-10)


def test_new_opportunity_decomposition_and_batch_independence():
    y, z, u, r, _ = sample(150)
    fit = estimate_response(y[:100], z[:100], u[:100], r[:100])
    result = apply_response(fit, z[100:], u[100:], r[100:], arm="control_function")
    np.testing.assert_allclose(
        result["response"], result["profile"] + result["common_input"]
    )
    single = apply_response(
        fit, z[100:101], u[100:101], r[100:101], arm="control_function"
    )
    np.testing.assert_allclose(single["response"], result["response"][:1])


def test_missing_opportunity_is_not_zero_personality():
    result = opportunity_profile(
        [[0.8, 0.2], [0.4, 0.3]], [[True, False], [True, False]]
    )
    assert result["score"][0] == pytest.approx(0.6)
    assert np.isnan(result["score"][1])
    with pytest.raises(ValueError):
        designs([1, 2], [1], [0, 0])


def test_hac_matches_explicit_bartlett_kernel():
    rng = np.random.default_rng(99)
    g = rng.normal(size=(20, 3))
    distance = abs(np.arange(20)[:, None] - np.arange(20))
    kernel = np.maximum(1 - distance / 5, 0)
    np.testing.assert_allclose(hac_meat(g, 4), g.T @ kernel @ g, atol=1e-10)


def test_public_replies_have_context_ambiguous_pair():
    from pathlib import Path

    from scripts.run_grain_interaction_response import public_bank
    from scripts.run_grain_query_moment_operator import read

    root = Path(__file__).resolve().parents[1]
    bank = public_bank(read(root / "configs/grain_interaction_response.json"))
    assert bank.case_id.nunique() == len(bank) == 72
    assert set(bank.act) == {"accept", "reject", "help", "repair", "past", "future"}
    assert sum(bank.reply == "Yes.") == 6
