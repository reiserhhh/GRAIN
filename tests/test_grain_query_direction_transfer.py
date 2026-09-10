#!/usr/bin/env python3
"""Role-bound transfer, unchanged diagonal and geometry/penalty distinctions."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_direction_transfer import (
    check_roles,
    fit_borrowed,
    subspace_geometry,
)
from grain_research.vector_mechanisms.query_residual_input import (
    fit_extension,
    predict_extension,
)
from tests.test_grain_query_residual_input import world


def test_identical_donor_reproduces_diagonal_and_keeps_recipient_output():
    hs, gs, js, base = world()
    diagonal, _ = fit_extension(hs, gs, js, base)
    role = {"held": 3, "source": [0, 1, 2]}
    donor = dict(diagonal)
    donor["basis"] = diagonal["basis"][::-1].copy()
    result, _ = fit_borrowed(hs, gs, js, base, donor, role, role, [0, 1, 2])
    np.testing.assert_array_equal(result["basis"], base["basis"])
    np.testing.assert_array_equal(result["aug_weights"], diagonal["aug_weights"])
    np.testing.assert_array_equal(
        predict_extension(hs[0], gs[0], result),
        predict_extension(hs[0], gs[0], diagonal),
    )


def test_signed_donor_basis_preserves_predictions():
    hs, gs, js, base = world()
    diagonal, _ = fit_extension(hs, gs, js, base)
    donor = dict(diagonal) | {"extra_basis": -diagonal["extra_basis"]}
    role = {"held": 3, "source": [0, 1, 2]}
    result, _ = fit_borrowed(hs, gs, js, base, donor, role, role, [0, 1, 2])
    np.testing.assert_allclose(
        predict_extension(hs[0], gs[0], result),
        predict_extension(hs[0], gs[0], diagonal),
        atol=1e-12,
    )


def test_roles_exclude_held_and_bind_donor():
    role = {"held": 3, "source": [0, 1, 2]}
    check_roles(role, role, [2, 0, 1])
    for bad in (
        {"held": 2, "source": [0, 1, 3]},
        {"held": 3, "source": [0, 1, 3]},
        {"held": 3, "source": [0, 0, 1]},
    ):
        with pytest.raises(ValueError):
            check_roles(role, bad, [0, 1, 2])


def test_subspace_angles_are_rotation_invariant_not_axis_correlation():
    a = np.eye(6)[:2]
    rotation = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    same = subspace_geometry(a, rotation @ a)
    assert same["projector_overlap"] == pytest.approx(1)
    assert same["angle_max_degrees"] < 2e-6
    perpendicular = subspace_geometry(a, np.eye(6)[2:4])
    assert perpendicular["projector_overlap"] == pytest.approx(0)
    assert perpendicular["angle_min_degrees"] == pytest.approx(90)
    with pytest.raises(ValueError):
        subspace_geometry(a, a * 2)


def test_rotation_requires_transforming_the_standardized_ridge_penalty():
    rng = np.random.default_rng(360)
    x = rng.normal(size=(100, 2)) * [1, 10]
    x -= x.mean(0)
    y = x @ [1, 0.3]
    r = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    sx = x.std(0)
    z = x / sx
    xr = x @ r
    sr = xr.std(0)
    zr = xr / sr
    transform = np.diag(sx) @ r @ np.diag(1 / sr)
    w = np.linalg.solve(z.T @ z + 10 * np.eye(2), z.T @ y)
    wr = np.linalg.solve(zr.T @ zr + 10 * transform.T @ transform, zr.T @ y)
    naive = np.linalg.solve(zr.T @ zr + 10 * np.eye(2), zr.T @ y)
    np.testing.assert_allclose(z @ w, zr @ wr, atol=1e-12)
    assert np.max(abs(z @ w - zr @ naive)) > 0.01
