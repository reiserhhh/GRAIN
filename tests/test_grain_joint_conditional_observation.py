#!/usr/bin/env python3
"""Mathematical checks for same-text multichannel conditional construction."""

import numpy as np
import pytest

from grain_research.joint_conditional_observation import (
    build_meter,
    calibrate,
    conditional_function,
    decoder,
    factorial,
    joint_probabilities,
    predict_function,
    read_events,
)


def fixture():
    """A full-rank meter with genuinely correlated observation errors."""
    rng = np.random.default_rng(103)
    b = rng.normal(size=6)
    l = rng.normal(size=(6, 3))
    raw = rng.normal(size=(6, 6))
    r = raw @ raw.T + np.eye(6)
    return b, l, r


def test_factorial_joint_moment_identity():
    m = np.array([[0.2, -0.1, 0.3], [0, 0, 0]])
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    p = joint_probabilities(m)
    np.testing.assert_allclose(p @ factorial(corners)[:, 1:], m, atol=1e-14)
    assert np.allclose(p.sum(axis=1), 1)


def test_impossible_moments_rejected():
    with pytest.raises(ValueError):
        joint_probabilities([1, 1, -1])


def test_calibration_full_rank_exact_means():
    rng = np.random.default_rng(43)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    poles = np.tile(np.repeat(corners, 20, axis=0), (2, 1))
    z = np.repeat([-1, 1], 80)
    b, l, _r = fixture()
    y = b + factorial(poles)[:, 1:] @ l.T + rng.normal(scale=0.1, size=(160, 6))
    model = calibrate(y, z, poles)
    meter = build_meter(model, 0.5, range(6))
    assert meter["rank"] == 3
    np.testing.assert_allclose(meter["resolution"], np.eye(3), atol=1e-12)


def test_affine_unit_invariance():
    b, l, r = fixture()
    d = decoder(b, l, r)
    transform = np.diag([0.0001, 2, -30000, 4, 0.5, 700])
    e = decoder(transform @ b, transform @ l, transform @ r @ transform.T)
    np.testing.assert_allclose(e["weights"] @ transform, d["weights"], atol=1e-11)
    np.testing.assert_allclose(
        e["working_covariance"], d["working_covariance"], atol=1e-11
    )


def test_duplicate_no_new_information():
    b, l, r = fixture()
    d = decoder(b, l, r)
    duplicate = np.vstack([np.eye(6), np.eye(6)[2]])
    e = decoder(duplicate @ b, duplicate @ l, duplicate @ r @ duplicate.T)
    np.testing.assert_allclose(e["weights"] @ duplicate, d["weights"], atol=1e-12)
    np.testing.assert_allclose(e["information"], d["information"], atol=1e-12)


def test_rank_deficiency_not_hidden():
    d = decoder(np.zeros(2), np.array([[1, 0, 0], [0, 1, 0]]), np.eye(2))
    assert d["rank"] == 2
    np.testing.assert_allclose(d["resolution"], np.diag([1, 1, 0]))


def test_conditional_function_keeps_level_and_change():
    a = np.array([[0.2, -0.1, 0.03]])
    b = np.array([[0.1, 0.04, -0.02]])
    y = np.repeat(np.stack([a - b, a + b], axis=1)[:, :, None, :], 4, axis=2)
    f = conditional_function(y)
    np.testing.assert_allclose(f["level"], a)
    np.testing.assert_allclose(f["slope"], b)
    np.testing.assert_allclose(predict_function(f, 0.5), a + 0.5 * b)
    np.testing.assert_allclose(conditional_function(y, mode="level")["slope"], 0)
    np.testing.assert_allclose(conditional_function(y, mode="slope")["level"], 0)


def test_shared_channel_noise_not_independent():
    l = np.ones((2, 1)) @ np.array([[1, 0, 0]])
    r = np.array([[1, 0.9], [0.9, 1]])
    d = decoder(np.zeros(2), l, r)
    naive = decoder(np.zeros(2), l, np.eye(2))
    assert d["working_covariance"][0, 0] > 0.9
    assert naive["working_covariance"][0, 0] == pytest.approx(0.5)


def test_current_rows_cannot_change_decoder():
    b, l, r = fixture()
    d = decoder(b, l, r)
    d["channels"] = np.arange(6)
    old = d["weights"].copy()
    read_events(np.ones((3, 6)), d)
    np.testing.assert_array_equal(d["weights"], old)


def test_equal_marginals_do_not_identify_event_pairing():
    """A same-event relation is absent from either marginal, even without noise."""
    s = np.array([-1, -1, 1, 1])
    t = s.copy()
    shifted = np.roll(t, 1)
    np.testing.assert_equal([s.mean(), t.mean(), shifted.mean()], [0, 0, 0])
    assert (s * t).mean() == 1
    assert (s * shifted).mean() == 0
    assert s.mean() * t.mean() == 0
    marginal = decoder(np.zeros(2), np.eye(3)[:2], np.eye(2))
    paired = decoder(np.zeros(3), np.eye(3), np.eye(3))
    assert marginal["rank"] == 2
    assert paired["rank"] == 3
    np.testing.assert_allclose(paired["resolution"], np.eye(3))


def test_full_rank_can_still_amplify_observation_error():
    """Invertibility is not adequate observation strength for every target."""
    strong = decoder(np.zeros(3), np.eye(3), np.eye(3))
    weak = decoder(np.zeros(3), np.diag([1, 1, 1e-4]), np.eye(3))
    assert strong["rank"] == weak["rank"] == 3
    np.testing.assert_allclose(weak["resolution"], np.eye(3))
    assert weak["working_covariance"][2, 2] == pytest.approx(1e8)
    assert strong["working_covariance"][2, 2] == pytest.approx(1)


def test_condition_dependent_meter_drift_mixes_level_and_response():
    """Different endpoint readout maps mix the two parts of one person function."""
    rng = np.random.default_rng(402)
    a, b, d_minus, d_plus = rng.normal(size=(4, 3))
    c_minus, c_plus = rng.normal(size=(2, 3, 3))
    minus = c_minus @ (a - b) + d_minus
    plus = c_plus @ (a + b) + d_plus
    same = (c_minus + c_plus) / 2
    cross = (c_plus - c_minus) / 2
    np.testing.assert_allclose(
        (minus + plus) / 2, same @ a + cross @ b + (d_minus + d_plus) / 2
    )
    np.testing.assert_allclose(
        (plus - minus) / 2, cross @ a + same @ b + (d_plus - d_minus) / 2
    )


def test_new_condition_covariance_keeps_level_response_cross_term():
    """Independent endpoint samples still induce correlated level/slope estimates."""
    rng = np.random.default_rng(207)
    events = rng.normal(size=(7, 2, 32, 3))
    events[:, 1] *= 3
    fit = conditional_function(events)
    z = 0.5
    propagated = (
        fit["level_covariance"]
        + z * z * fit["slope_covariance"]
        + 2 * z * fit["level_slope_covariance"]
    )
    direct = ((1 - z) / 2) ** 2 * fit["material_mean_covariance"][:, 0] + (
        (1 + z) / 2
    ) ** 2 * fit["material_mean_covariance"][:, 1]
    np.testing.assert_allclose(propagated, direct, atol=1e-14)
