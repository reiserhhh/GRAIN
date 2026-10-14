#!/usr/bin/env python3
"""HM29 support algebra, physical patch units and source-only action witnesses."""

import numpy as np
import pytest

from grain_research.construction import math_position_support as P
from grain_research.construction import math_source_field as S


def test_source_complement_is_exact_in_declared_metric():
    ds = P.directions(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    support = S.common_covector([np.ones((1, 3)), np.ones((2, 3))])["support"]
    for arm in P.ARMS[:2]:
        m = S.moment_map(4) @ ds[arm]
        np.testing.assert_allclose(support @ m, 0, atol=3e-14)
        assert np.linalg.norm(m) > 1
    m = S.moment_map(4) @ ds["known_mean"]
    np.testing.assert_allclose(support @ m, m, atol=3e-14)


def test_toy_vanishing_polynomial_is_not_pure_complement():
    t = np.linspace(-1, 1, 4)
    support = S.common_covector([np.ones((1, 1)), np.ones((2, 1))])["support"]
    m = S.moment_map(4) @ (t**3 - t)[:, None]
    assert np.linalg.norm(support @ m) > 0.01


def test_known_common_field_cannot_fix_missing_coefficient():
    g = [np.array([[1.0], [2.0]]), np.array([[4.0]])]
    old = S.common_covector(g)["covector"]
    delta = P.directions(np.ones(1), np.ones(1))["unseen_source"]
    m = S.moment_map(4) @ delta
    assert abs(S.contract(old, m)) < 1e-13
    other = old + np.array([0.0, -1.0, 0.0, 1.0])[:, None]
    assert S.contract(other, m) == pytest.approx(2.0)


def test_new_four_token_constraint_identifies_constant_field():
    true = np.arange(12, dtype=float).reshape(4, 3)
    gradient = S.moment_map(4).T @ true
    field = S.common_covector([gradient] * 8)
    assert field["rank"] == 4
    np.testing.assert_allclose(field["covector"], true, atol=2e-13)


def test_physical_patches_keep_casting_asymmetry():
    h = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    d = P.directions(np.ones(3), np.ones(3))["unseen_source"]
    r = P.physical_patches(h, d)
    np.testing.assert_array_equal(r["delta_plus"], r["plus"].astype(float) - h)
    np.testing.assert_allclose(np.linalg.norm(r["intended"]), 0.01 * np.linalg.norm(h))
    np.testing.assert_array_equal(
        r["half_delta"], (r["delta_plus"] - r["delta_minus"]) / 2
    )


def test_symmetric_action_distinguishes_curvature():
    a = P.contrast(1.0, 1.21, 0.81)
    assert a["odd_action"] == pytest.approx(0.2)
    assert a["even_change"] == pytest.approx(0.01)


def test_all_materials_have_supplied_self_ownership():
    for i in range(16):
        text = P.material(i, P.PHRASES[i])
        assert text.count("I felt ") == 1
        assert "I judged AI systems generally useful" in text
    with pytest.raises(ValueError):
        P.material(16, P.PHRASES[0])


def test_zero_or_mismatched_hidden_direction_rejected():
    with pytest.raises(ValueError):
        P.directions(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        P.directions(np.ones(2), np.ones(3))


def test_incorrect_span_shape_or_radius_rejected():
    with pytest.raises(ValueError):
        P.physical_patches(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        P.physical_patches(np.ones((4, 3)), np.ones((4, 3)), 0)


def test_proper_frame_and_metric_preserve_complement_action():
    r = np.diag([1.0, 2.0, 3.0, 4.0])
    n = np.array([0.0, -1.0, 0.0, 1.0])
    oldsupport = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    metric = np.linalg.inv(r).T @ np.linalg.inv(r)
    np.testing.assert_allclose((r @ oldsupport).T @ metric @ (r @ n), 0)
    coeff = np.arange(4.0)
    assert coeff @ n == pytest.approx(np.linalg.solve(r.T, coeff) @ (r @ n))
