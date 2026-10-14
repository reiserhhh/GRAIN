#!/usr/bin/env python3
"""Mechanism nulls and arithmetic checks for the fixed HM108 observation account."""

import numpy as np
import pytest

from grain_research.construction import math_native_readout_location as M
from grain_research.construction.math_joint_reading import BASIS
from grain_research.construction.math_readout_curvature import link_derivatives


def test_additive_logits_can_have_joint_readout_without_joint_field():
    base = np.linspace(-0.4, 0.4, 10)
    x = np.eye(10)[9] * 0.7
    y = np.eye(10)[9] * 0.8
    z = base + BASIS[1, :, None] * x + BASIS[2, :, None] * y
    result = M.decompose(z)
    assert np.linalg.norm(result["natural"][3]) < 1e-12
    assert np.all(abs(result["characters"]["additive_readout"][3]) > 1e-4)
    assert np.max(abs(result["characters"]["joint_field_lift"])) < 1e-12
    assert M.integral_account(z)["error"] < 1e-12


def test_pure_joint_field_and_no_joint_null_are_separable():
    z = BASIS[3, :, None] * np.eye(10)[9] * 0.4
    result = M.decompose(z)
    assert np.max(abs(result["characters"]["additive_readout"][3])) < 1e-12
    assert np.all(abs(result["characters"]["joint_field_lift"][3]) > 1e-4)
    assert M.integral_account(z)["error"] < 1e-12
    null = M.decompose(np.zeros((4, 10)))
    assert np.max(abs(null["characters"]["native"][1:])) < 1e-12


def test_gauge_and_corner_relabelling_do_not_change_the_account():
    z = np.random.default_rng(8).normal(size=(4, 10)) / 3
    baseline = M.decompose(z)
    shifted = M.decompose(z + np.array([1, -2, 4, -3])[:, None])
    swapped = M.decompose(z[[1, 0, 3, 2]])
    for part in M.PARTS:
        assert np.allclose(baseline["characters"][part], shifted["characters"][part])
        expected = baseline["characters"][part] * np.array([1, -1, 1, -1])[:, None]
        assert np.allclose(swapped["characters"][part], expected)
    assert baseline["error"] < 1e-12


def test_derivatives_match_existing_expectation_and_finite_differences():
    z = np.random.default_rng(19).normal(size=10)
    g, h = M.derivatives(z)
    old = link_derivatives(z, M.VALUES)
    assert np.allclose(g[0], old[2])
    assert np.allclose(h[0], old[3])
    epsilon = 1e-5
    numeric = np.stack(
        [
            (M.derivatives(z + epsilon * e)[0] - M.derivatives(z - epsilon * e)[0])
            / (2 * epsilon)
            for e in np.eye(10)
        ],
        axis=-1,
    )
    assert np.max(abs(numeric - h)) < 1e-9
    assert np.max(abs(h @ np.ones(10))) < 1e-12
    assert np.linalg.eigvalsh(h[0]).min() < 0


def test_positive_author_composition_and_invalid_inputs():
    result = M.decompose(np.random.default_rng(3).normal(size=(4, 10)))
    for law in ([0.25] * 4, [0.4, 0.1, 0.2, 0.3], [0.375, 0.125, 0.125, 0.375]):
        _, error = M.author_account(result, law)
        assert error < 1e-12
    with pytest.raises(ValueError):
        M.author_account(result, [-1, 1, 1, 0])
    with pytest.raises(ValueError):
        M.decompose(np.zeros((3, 10)))
    with pytest.raises(ValueError):
        M.readout(np.full(10, np.nan))


def test_signed_cancellation_is_not_an_information_fraction():
    a, b = np.array([2.0, -2.0]), np.array([-1.9, 1.9])
    account = M.signed_account(a + b, a, b)
    assert account["opposite_sign_count"] == 2
    assert account["additive_signed_alignment"] > 1
    assert account["lift_signed_alignment"] < 0
    assert account["second_moment_error"] < 1e-12
    zero = M.signed_account(np.zeros(2), a, -a)
    assert zero["additive_signed_alignment"] is None
