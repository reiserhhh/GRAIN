#!/usr/bin/env python3
"""Finite witnesses for covariance pairing, physical units and signed transport."""

import numpy as np
import pytest

from grain_research.construction import math_active_reading_pairing as P


def test_constant_coefficients_have_no_coupling():
    t = np.array([[1, 2], [1, 2]])
    e = np.array([[3, 4], [-1, 8]])
    pair, words = P.decompose(t, e, np.sum(t * e, axis=1), [1, 2])
    assert pair["coupling"] == 0 and pair["error"] == 0
    assert all(r["error_identity_error"] == 0 for r in words)


def test_orthogonal_differences_are_not_a_scalar_coupling():
    pair, _ = P.decompose([[1, 0], [-1, 0]], [[0, 2], [0, -2]], [0, 0], [0, 0])
    assert pair["coupling"] == 0 and pair["covariance_frobenius"] == 2
    assert pair["coupling_direction_cosine"] == 0


def test_pairing_only_world_cannot_factor_through_means():
    pair, words = P.decompose([[2], [-2]], [[3], [-3]], [6, 6], [0])
    assert pair["common"] == 0 and pair["coupling"] == 6
    assert pair["error"] == 6 and pair["error_identity_error"] == 0
    assert all(r["error"] == 6 for r in words)


def test_negative_coupling_and_carrier_cancel_without_deleting_either():
    pair, words = P.decompose([[2], [-2]], [[-3], [3]], [0, 0], [0])
    assert pair["coupling"] == -6 and pair["carrier"] == 6
    metric = P.component_metrics([pair], P.PAIR_COMPONENTS)
    assert metric["relative_error"] is None and metric["error_rms"] == 0
    assert metric["carrier__coupling_cross"] == -72
    assert metric["second_moment_identity_error"] == 0
    assert all(r["error_identity_error"] == 0 for r in words)


def test_unequal_H_norms_preserve_each_physical_action():
    a, z, n = np.array([[2, 4], [6, 8]]), np.array([3, 5]), np.array([2, 7])
    theta, eta = P.physical_pair(a, z, n)
    np.testing.assert_allclose(np.sum(theta * eta, axis=1), a @ z)


def test_orthogonal_coframe_change_preserves_pairing_and_transport():
    rng = np.random.default_rng(38)
    theta, eta, ref = (
        rng.normal(size=(2, 7)),
        rng.normal(size=(2, 7)),
        rng.normal(size=7),
    )
    full = np.sum(theta * eta, axis=1) + [0.2, -0.3]
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    p, w = P.decompose(theta, eta, full, ref)
    pr, wr = P.decompose(theta @ q, eta @ q, full, ref @ q)
    for k in p:
        np.testing.assert_allclose(p[k], pr[k], atol=1e-14)
    for a, b in zip(w, wr, strict=True):
        for k in a:
            np.testing.assert_allclose(a[k], b[k], atol=1e-14)


def test_mean_transport_and_wording_contrasts_both_restore_errors():
    theta, eta = np.array([[3, 1], [1, -1]]), np.array([[2, 4], [4, 2]])
    full = np.sum(theta * eta, axis=1) + [1, -2]
    p, w = P.decompose(theta, eta, full, [1, 0])
    assert p["shared_transport"] != 0
    assert p["error_identity_error"] == 0
    assert all(
        r["error_identity_error"] == 0 and r["supported_identity_error"] == 0 for r in w
    )
    assert (
        P.component_metrics(w, P.WORD_COMPONENTS)["second_moment_identity_error"] == 0
    )


def test_nonzero_mean_vector_shift_can_have_zero_scalar_transport():
    theta, eta = np.array([[3, 1], [1, -1]]), np.array([[2, 4], [4, 2]])
    ref = np.array([1, 1])
    assert np.linalg.norm(theta.mean(axis=0) - ref) > 0
    pair, _ = P.decompose(theta, eta, np.sum(theta * eta, axis=1), ref)
    assert pair["shared_transport"] == 0


def test_bad_units_and_nonfinite_observations_rejected():
    with pytest.raises(ValueError):
        P.physical_pair([[1], [2]], [3], [1, 0])
    with pytest.raises(ValueError):
        P.decompose([[np.nan], [0]], [[0], [0]], [0, 0], [0])


def test_integrate_node_pairing_not_pair_integrated_marginals():
    theta = [
        np.array([[1.0], [-1.0]]),
        np.array([[2.0], [-2.0]]),
        np.array([[3.0], [-3.0]]),
    ]
    eta = [a.copy() for a in theta]
    weights = [1 / 12, 1 / 3, 1 / 12]
    values = [
        P.decompose(t, e, np.sum(t * e, axis=1), [0])[0]["coupling"]
        for t, e in zip(theta, eta, strict=True)
    ]
    integral = sum(w * v for w, v in zip(weights, values, strict=True))
    integrated_theta = sum(w * v for w, v in zip(weights, theta, strict=True))
    integrated_eta = sum(w * v for w, v in zip(weights, eta, strict=True))
    wrong = P.decompose(integrated_theta, integrated_eta, [0, 0], [0])[0]["coupling"]
    assert not np.isclose(integral, wrong)
