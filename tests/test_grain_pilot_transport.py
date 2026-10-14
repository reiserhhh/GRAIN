#!/usr/bin/env python3
"""Independent formula and boundary checks for pilot-based measurement construction."""

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from grain_research.pilot_transport import (
    calibration,
    covariance_parameters,
    fit_pilot,
    reml_objective,
    restore_units,
    score_batch,
    total_uncertainty,
)


def fixture():
    """Synthetic observations with a declared, full-rank common schedule."""
    rng = np.random.default_rng(42)
    t = np.r_[0, np.cumsum(np.resize([0.2, 0.8, 1.5, 3.5], 15))]
    z = np.resize([-1, -1, 1, 1], len(t)).astype(float)
    u = np.resize([-1, 1, -1, 1], len(t)).astype(float)
    x = np.c_[np.ones(len(t)), z, u]
    beta = rng.normal([0.1, 0.2, -0.1], [0.18, 0.12, 0.14], (64, 3))
    state = rng.multivariate_normal(
        np.zeros(len(t)), 0.03 * np.exp(-0.2 * abs(t[:, None] - t)), len(beta)
    )
    repeated = (beta @ x.T + state)[:, :, None] + rng.normal(
        0, 0.2, (len(beta), len(t), 4)
    )
    return repeated, z, u, t


def test_reml_gradient_matches_finite_difference():
    y, z, u, t = fixture()
    x = np.c_[np.ones(len(t)), z, u]
    theta = np.array([-1.6, 0.03, -2, -0.01, 0.04, -1.9, -3, -1.5])
    args = (y.mean(axis=2), x, t, 0.01)
    value, grad = reml_objective(theta, *args)
    numeric = approx_derivative(lambda a: reml_objective(a, *args)[0], theta).ravel()
    assert np.isfinite(value)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-5)


def test_reml_matches_explicit_stacked_covariance():
    y, z, u, t = fixture()
    y = y[:8].mean(axis=2)
    x = np.c_[np.ones(len(t)), z, u]
    theta = np.array([-1.6, 0.03, -2, -0.01, 0.04, -1.9, -3, -1.5])
    value, mean, k = reml_objective(theta, y, x, t, 0.01, gradient=False)
    omega = np.kron(np.eye(len(y)), k)
    xx = np.tile(x, (len(y), 1))
    inv = np.linalg.inv(omega)
    m = np.linalg.solve(xx.T @ inv @ xx, xx.T @ inv @ y.ravel())
    residual = y.ravel() - xx @ m
    expected = 0.5 * (
        2 * np.log(np.diag(np.linalg.cholesky(omega))).sum()
        + 2 * np.log(np.diag(np.linalg.cholesky(xx.T @ inv @ xx))).sum()
        + residual @ inv @ residual
    )
    np.testing.assert_allclose(value, expected, atol=1e-9)
    np.testing.assert_allclose(mean, m, atol=1e-10)


def test_calibration_needs_paired_units_and_condition():
    rng = np.random.default_rng(10)
    a, z = rng.normal(size=100), np.resize([-1.0, 1.0], 100)
    b = 0.4 + 0.2 * z + (1.8 + 0.6 * z) * a
    local = calibration(a, b, z, conditional=True)
    global_map = calibration(a, b, z)
    np.testing.assert_allclose(restore_units(b, z, local), a, atol=1e-12)
    assert np.mean((restore_units(b, z, global_map) - a) ** 2) > 0.01
    with pytest.raises(ValueError):
        calibration(np.ones(100), b, z)
    with pytest.raises(ValueError):
        restore_units(b, np.full(100, -10.0), local)


def test_pilot_and_batch_scoring_without_evaluation_fit():
    y, z, u, t = fixture()
    fitted = fit_pilot(y[:48], z, u, t)
    assert fitted["rate"] > 0 and fitted["stationary"] > 0
    assert fitted["success"], fitted["message"]
    targets = ["level", "condition", "current_state", "conditional_profile"]
    scores, var = score_batch(y[48:].mean(axis=2), z, u, t, fitted, targets)
    single, single_var = score_batch(y[48:49].mean(axis=2), z, u, t, fitted, targets)
    np.testing.assert_allclose(scores[:1], single, atol=1e-12)
    np.testing.assert_allclose(var[:1], single_var, atol=1e-12)
    assert np.isfinite(scores).all() and np.all(var >= 0)


def test_affine_model_equivariance_for_level_and_state():
    y, z, u, t = fixture()
    fitted = fit_pilot(y, z, u, t)
    c, d = 0.4, 1.8
    moved = dict(fitted)
    moved["mean"] = d * fitted["mean"] + [c, 0, 0]
    moved["covariance"] = d**2 * fitted["covariance"]
    moved["stationary"] = d**2 * fitted["stationary"]
    moved["noise"] = d**2 * fitted["noise"]
    targets = ["level", "condition", "current_state", "current_response"]
    old, v = score_batch(y[:3].mean(axis=2), z, u, t, fitted, targets)
    new, vv = score_batch(c + d * y[:3].mean(axis=2), z, u, t, moved, targets)
    np.testing.assert_allclose(new, d * old + [c, 0, 0, c], atol=1e-10)
    np.testing.assert_allclose(vv, d**2 * v, atol=1e-10)


def test_total_variance_keeps_construction_component():
    p = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    v = np.full_like(p, 0.5)
    result = total_uncertainty(p, v)
    np.testing.assert_allclose(result["variance"], 0.5 + p.var(axis=0, ddof=1))
    assert np.all(result["between"] > 0)


def test_rank_and_replicate_boundaries():
    y, z, u, t = fixture()
    with pytest.raises(ValueError):
        fit_pilot(y, z, np.ones(len(t)), t)
    with pytest.raises(ValueError):
        fit_pilot(y[:, :, :1], z, u, t)
    with pytest.raises(ValueError):
        fit_pilot(y, z, u, np.zeros(len(t)))
    v, p, rate, _ = covariance_parameters(np.zeros(8))
    assert np.linalg.eigvalsh(v).min() > 0 and p > 0 and rate > 0


def test_condition_meter_mixes_level_response_and_partner_interaction():
    """For z squared=1, meter effects alias profile and add a partner cross-term."""
    rng = np.random.default_rng(413)
    a, b, c, u, s = rng.normal(size=(5, 100))
    z = np.resize([-1.0, 1.0], 100)
    c0, c1, d0, d1 = 0.6, 0.15, 1.8, 0.6
    direct = c0 + c1 * z + (d0 + d1 * z) * (a + b * z + c * u + s)
    decomposed = (
        (c0 + d0 * a + d1 * b)
        + (c1 + d0 * b + d1 * a) * z
        + d0 * c * u
        + d1 * c * z * u
        + (d0 + d1 * z) * s
    )
    np.testing.assert_allclose(direct, decomposed, atol=1e-12)


def test_independent_joint_score_audit():
    """Audit's direct normal equations and production kernel give the same target."""
    from scripts.audit_grain_pilot_transport import independent_scores

    y, z, u, t = fixture()
    fitted = fit_pilot(y[:48], z, u, t)
    targets = [
        "level",
        "condition",
        "partner",
        "current_state",
        "current_response",
        "window_state",
        "conditional_profile",
    ]
    values = y[48:].mean(axis=2)
    p, v = score_batch(values, z, u, t, fitted, targets)
    pp, vv = independent_scores(values, z, u, t, fitted, targets)
    np.testing.assert_allclose(p, pp, atol=1e-10)
    np.testing.assert_allclose(v, vv, atol=1e-10)
