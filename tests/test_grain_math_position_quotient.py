#!/usr/bin/env python3
"""Exact aliases, weighted gradient laws and missing-support witnesses."""

from fractions import Fraction as F

import numpy as np
import pytest

from grain_research.construction import math_position_quotient as Q
from grain_research.construction import math_source_field as S


def test_exact_source_ideal_and_cubic_reduction():
    cert = Q.certificate([1, 2])
    assert cert["positions"] == [F(-1), F(0), F(1)]
    assert cert["polynomial"] == [F(0), F(-1), F(0), F(1)]
    assert (cert["rank"], cert["nullity"]) == (3, 1)
    assert list(cert["reduction"][:, 3]) == [0, 1, 0]


def test_four_positions_remove_truncated_cubic_alias():
    assert Q.certificate([1, 2, 3])["rank"] == 3
    cert = Q.certificate([1, 2, 4])
    assert cert["rank"] == 4 and cert["nullity"] == 0
    assert len(cert["positions"]) == 5


def test_null_witness_is_zero_then_nonzero_exactly():
    p = Q.certificate([1, 2])["polynomial"]
    assert all(Q.exact_witness(t, p)["single_action"] == 0 for t in (1, 2, 3))
    assert Q.exact_witness(4, p)["single_action"] == F(32, 729)


def test_grouped_loss_matches_joint_constraints_with_repeats():
    rng = np.random.default_rng(2801)
    gradients = [rng.normal(size=(t, 3)) for t in (1, 1, 2, 3, 2)]
    field = Q.fit_node_field(gradients)
    joint = S.common_covector(gradients)["covector"]
    for t in (1, 2, 3):
        delta = rng.normal(size=(t, 3))
        expected = S.contract(joint, S.moment_map(t) @ delta)
        assert Q.node_action(field, delta) == pytest.approx(expected, abs=2e-13)
    assert field["weights"][0] == F(1, 5 * 8) * 2 + F(1, 5 * 27)


def test_different_representatives_have_equal_supported_actions():
    rng = np.random.default_rng(2802)
    gs = [rng.normal(size=(t, 2)) for t in (1, 2)]
    field = Q.fit_node_field(gs)
    assert (
        np.linalg.norm(field["coefficients"] - S.common_covector(gs)["covector"]) > 0.1
    )
    for t in (1, 2, 3):
        d = rng.normal(size=(t, 2))
        assert Q.node_action(field, d) == pytest.approx(
            S.contract(field["coefficients"], S.moment_map(t) @ d), abs=2e-13
        )


def test_interpolation_recovers_quadratic_node_values():
    coef = np.array([[2.0, 1.0], [3.0, 4.0], [5.0, -2.0], [0.0, 0.0]])
    pts = [F(-1), F(0), F(1)]
    values = np.array(
        [
            [sum(float(t**k) * coef[k, j] for k in range(4)) for j in range(2)]
            for t in pts
        ]
    )
    np.testing.assert_allclose(Q.interpolate(pts, values), coef)


def test_new_position_is_not_filled_with_false_observed_value():
    field = Q.fit_node_field([np.ones((1, 2)), np.ones((2, 2))])
    with pytest.raises(ValueError, match="not source-identified"):
        Q.node_action(field, np.ones((4, 2)))
    with pytest.raises(ValueError, match="exceed"):
        Q.fit_node_field([np.ones((1, 2)), np.ones((4, 2))])


def test_duplicate_sources_change_weights_not_exact_rank():
    assert Q.certificate([1, 2] * 100)["rank"] == 3
    assert Q.certificate([1])["rank"] == 1


def test_invalid_position_input_is_rejected():
    for invalid in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            Q.positions(invalid)
    with pytest.raises(ValueError):
        Q.certificate([])


def test_interpolation_requires_distinct_finite_points():
    with pytest.raises(ValueError):
        Q.interpolate([F(0), F(0)], np.ones((2, 1)))
    with pytest.raises(ValueError):
        Q.fit_node_field([np.array([[np.nan]])])
