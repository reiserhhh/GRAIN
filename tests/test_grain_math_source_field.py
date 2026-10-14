#!/usr/bin/env python3
"""Synthetic support/gauge/constant-field witnesses for HM27."""

import numpy as np
import pytest

from grain_research.construction import math_source_field as S


def test_observation_is_injective_in_each_token_stratum():
    rng = np.random.default_rng(27)
    for count in range(1, 5):
        gradient = rng.normal(size=(count, 3))
        result = S.extension(gradient)
        assert result["rank"] == count
        np.testing.assert_allclose(
            S.moment_map(count).T @ result["covector"], gradient, atol=2e-14
        )


def test_covector_action_survives_proper_frame_transport():
    rng = np.random.default_rng(2701)
    frame = np.diag([1, 2, 3, 4])
    c, dx = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    assert S.contract(c, dx) == pytest.approx(
        S.contract(np.linalg.solve(frame.T, c), frame @ dx), abs=2e-14
    )


def test_new_minimum_norm_extension_is_not_covector_transport():
    frame = np.diag([1, 2, 3, 4])
    gradient = np.array([[2.0], [4.0]])
    old = S.extension(gradient)["covector"]
    new = S.extension(gradient, frame)["covector"]
    source_dx = S.moment_map(2) @ np.array([[0.4], [0.8]])
    assert S.contract(old, source_dx) == pytest.approx(
        S.contract(new, frame @ source_dx)
    )
    cold_dx = S.moment_map(1) @ np.ones((1, 1))
    assert abs(S.contract(old, cold_dx) - S.contract(new, frame @ cold_dx)) > 0.1


def test_joint_constraints_recover_a_constant_field():
    rng = np.random.default_rng(2702)
    true = rng.normal(size=(4, 2))
    gradients = [S.moment_map(t).T @ true for t in (1, 2, 4)]
    joint = S.common_covector(gradients)
    assert joint["rank"] == 4
    np.testing.assert_allclose(joint["covector"], true, atol=1e-13)
    averaged = np.mean([S.extension(g)["covector"] for g in gradients], axis=0)
    assert np.linalg.norm(averaged - true) > 0.1


def test_supported_cold_actions_unique_even_with_rank_deficiency():
    true = np.arange(8, dtype=float).reshape(4, 2)
    gradients = [S.moment_map(t).T @ true for t in (1, 2)]
    frame = np.diag([1, 2, 3, 4])
    old, new = S.common_covector(gradients), S.common_covector(gradients, frame)
    assert old["rank"] == 3
    for count in (1, 2):
        dx = S.moment_map(count) @ np.ones((count, 2))
        assert S.contract(old["covector"], dx) == pytest.approx(
            S.contract(new["covector"], frame @ dx), abs=1e-13
        )


def test_unseen_support_is_not_identified():
    result = S.common_covector([np.ones((1, 1))])
    assert result["rank"] == 1
    cold = S.moment_map(2) @ np.array([[0.0], [1.0]])
    assert np.linalg.norm(cold - result["support"] @ cold) > 0.1


def test_curved_field_start_differs_from_path_mean():
    start = np.array([[0.0]])
    delta = np.ones((1, 1))
    integral = np.array([[1.0]])
    assert S.contract(start, delta) == 0
    assert S.contract(integral, delta) == 1


def test_joint_residual_reveals_nonshared_gradients():
    result = S.common_covector([np.ones((2, 1)), -np.ones((2, 1))])
    assert result["weighted_residual_rms"] == pytest.approx(1)


def test_signed_covariance_reconstructs_native():
    rng = np.random.default_rng(2703)
    y, z, a, b = rng.normal(size=(4, 8))
    r = S.paired_covariance(y, z, a, b)
    assert r["native"] == pytest.approx(
        sum(r[k] for k in ("PP", "PR", "RP", "RR")), abs=1e-14
    )


def test_invalid_inputs_and_no_source():
    with pytest.raises(ValueError):
        S.moment_map(5)
    with pytest.raises(ValueError):
        S.common_covector([])
    with pytest.raises(ValueError):
        S.extension(np.ones((2, 3)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        S.contract(np.ones((2, 3)), np.ones((4, 3)))
