#!/usr/bin/env python3
"""Independent latent-state checks for contextual text measurement targets."""

import numpy as np
import pytest
from scipy.linalg import expm

from grain_research.temporal_measurement import (
    interaction_pairs,
    observation_design,
    posterior,
    score_target,
    state_covariance,
    target_vector,
)


def fixture():
    """Known two-channel process with independent noisy observations."""
    rng = np.random.default_rng(814)
    times = np.array([0.0, 0.1, 0.4, 2.0, 4.0, 7.0])
    params = {
        "drift": np.array([[-0.3, 0.12], [-0.04, -0.2]]),
        "stationary": np.eye(2) * 0.1,
        "noise": np.array([[0.12, 0.02], [0.02, 0.15]]),
        "profile_mean": np.array([0.0, 0.0, 0.2] * 2),
        "profile_cov": np.eye(6) * 0.2,
    }
    y, z, u = rng.normal(size=(6, 2)), rng.normal(size=(6, 2)), rng.normal(size=6)
    return times, y, z, u, params


def kalman_final(times, y, z, u, params):
    """Independent recursive implementation; does not call batch covariance code."""
    q = y.shape[1]
    m = np.r_[params["profile_mean"], np.zeros(q)]
    p = np.zeros((4 * q, 4 * q))
    p[: 3 * q, : 3 * q] = params["profile_cov"]
    p[3 * q :, 3 * q :] = params["stationary"]
    for i, time in enumerate(times):
        if i:
            transition = expm(params["drift"] * (time - times[i - 1]))
            f = np.eye(4 * q)
            f[3 * q :, 3 * q :] = transition
            process = np.zeros_like(p)
            process[3 * q :, 3 * q :] = (
                params["stationary"] - transition @ params["stationary"] @ transition.T
            )
            m, p = f @ m, f @ p @ f.T + process
        h = np.zeros((q, 4 * q))
        for k in range(q):
            h[k, 3 * k : 3 * k + 3] = [1, z[i, k], u[i]]
            h[k, 3 * q + k] = 1
        observed = np.isfinite(y[i])
        h = h[observed]
        r = params["noise"][np.ix_(observed, observed)]
        gain = np.linalg.solve(h @ p @ h.T + r, h @ p).T
        m += gain @ (y[i, observed] - h @ m)
        p -= gain @ h @ p
    return m, p


def test_batch_matches_independent_kalman_endpoint():
    times, y, z, u, params = fixture()
    fit = posterior(times, y, z, u, **params)
    expected, cov = kalman_final(times, y, z, u, params)
    index = np.r_[np.arange(6), np.arange(len(fit["mean"]) - 2, len(fit["mean"]))]
    np.testing.assert_allclose(fit["mean"][index], expected, atol=1e-12)
    np.testing.assert_allclose(fit["covariance"][np.ix_(index, index)], cov, atol=1e-12)


def test_missing_evidence_remains_unobserved():
    times, y, z, u, params = fixture()
    y[2:4, 1] = np.nan
    fit = posterior(times, y, z, u, **params)
    assert fit["observed"].sum() == 10
    np.testing.assert_array_equal(fit["kernels"][:, [5, 7]], 0)
    expected, _ = kalman_final(times, y, z, u, params)
    np.testing.assert_allclose(fit["mean"][-2:], expected[-2:], atol=1e-12)


def test_time_unit_change_transports_drift():
    t, y, z, u, p = fixture()
    original = posterior(t, y, z, u, **p)
    changed = posterior(t * 60, y, z, u, **{**p, "drift": p["drift"] / 60})
    np.testing.assert_allclose(original["mean"], changed["mean"], atol=1e-12)
    for name in ("current_state", "window_state", "level", "conditional_profile"):
        x = score_target(original, target_vector(original, name, 0))
        yy = score_target(changed, target_vector(changed, name, 0))
        assert x["score"] == pytest.approx(yy["score"])


def test_target_kernel_reconstructs_score_and_window_preserves_constant():
    t, y, z, u, p = fixture()
    fit = posterior(t, y, z, u, **p)
    for name in (
        "level",
        "condition",
        "partner",
        "current_state",
        "current_response",
        "window_state",
        "window_response",
        "conditional_profile",
    ):
        w = target_vector(fit, name, 0, z=0.5, partner=-0.2)
        score = score_target(fit, w)
        assert score["score"] == pytest.approx(
            score["offset"] + score["kernel"] @ y.ravel()
        )
        assert score["lower"] <= score["score"] <= score["upper"]
    window = target_vector(fit, "window_state", 0)
    assert window[6::2].sum() == pytest.approx(1)
    assert not np.allclose(window[6::2], np.ones(len(t)) / len(t))


def test_constant_partner_identifies_sum_not_person_and_partner_separately():
    t, y, z, _, p = fixture()
    u = np.ones(len(t))
    fit = posterior(t, y, z, u, **p)
    assert np.linalg.matrix_rank(observation_design(z, u)) == 4
    difference = np.zeros(len(fit["mean"]))
    difference[0], difference[2] = 1, -1
    read = score_target(fit, difference)
    assert read["variance"] == pytest.approx(0.4)
    np.testing.assert_allclose(read["kernel"], 0, atol=1e-12)


def test_coupling_adds_cross_channel_observation_path():
    t, y, z, u, p = fixture()
    y[:, 1] = np.nan
    coupled = posterior(t, y, z, u, **p)
    uncoupled = posterior(t, y, z, u, **{**p, "drift": np.diag(np.diag(p["drift"]))})
    kc = score_target(coupled, target_vector(coupled, "current_state", 1))["kernel"]
    ku = score_target(uncoupled, target_vector(uncoupled, "current_state", 1))["kernel"]
    assert np.linalg.norm(kc) > 1e-5
    np.testing.assert_allclose(ku, 0, atol=1e-12)


def test_covariance_and_clock_constraints():
    t, y, z, u, p = fixture()
    assert (
        np.linalg.eigvalsh(state_covariance(t, p["drift"], p["stationary"])).min() > 0
    )
    with pytest.raises(ValueError, match="Strictly"):
        posterior([0, 0, 1, 2, 3, 4], y, z, u, **p)
    with pytest.raises(ValueError, match="Stable"):
        posterior(t, y, z, u, **{**p, "drift": np.eye(2)})
    with pytest.raises(ValueError, match="No observed"):
        posterior(t, y * np.nan, z, u, **p)


def test_pairs_never_cross_missing_turns_or_duplicate_roles():
    rows = [
        {"role": "assistant", "turn_sequence": 0},
        {"role": "user", "turn_sequence": 1},
        {"role": "assistant", "turn_sequence": 3},
        {"role": "user", "turn_sequence": 5},
    ]
    assert len(interaction_pairs(rows)) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        interaction_pairs(rows + rows[:1])


def test_portable_inventory_ignores_metadata_not_science():
    from scripts.audit_grain_temporal_measurement import scientific_files

    assert scientific_files({"scores.csv": "a", "._scores.csv": "b"}) == {
        "scores.csv": "a"
    }
