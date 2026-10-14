#!/usr/bin/env python3
"""Synthetic tests for target specificity versus generic and contextual effects."""

import numpy as np
import pytest

from grain_research.construction import math_target_bound_actions as M


def test_complete_cube_and_character_reconstruction():
    a = np.arange(8 * 6 * 3, dtype=float).reshape(8, 6, 3)
    c = M.coefficients(a)
    assert np.allclose(np.tensordot(M.SIGNS.T, c, axes=(1, 0)), a)
    assert np.allclose(M.SIGNS @ M.SIGNS.T, 8 * np.eye(8))


def test_query_independent_body_retains_shared_action():
    a = np.repeat(np.arange(24).reshape(8, 1, 3), 6, axis=1)
    c = M.coefficients(a)
    for term in c:
        s = M.query_account(term)
        assert np.allclose(s["varying"], 0)
        assert np.allclose(s["shared"], term[0])


def test_query_split_is_lossless_and_orthogonal():
    a = np.arange(30).reshape(6, 5) / 7
    s = M.query_account(a)
    assert s["energy_error"] < 1e-12
    assert np.allclose(s["shared"] + s["varying"], a)
    assert np.allclose(s["varying"].sum(axis=0), 0)


@pytest.mark.parametrize("kind", ("generic", "selective", "conditional", "role_only"))
@pytest.mark.parametrize("focal", (None, -1, 1))
def test_direct_and_coefficient_ownership_actions(kind, focal):
    values = M.toy_world(kind)
    assert np.allclose(
        M.ownership_action(M.coefficients(values), focal),
        M.direct_ownership_action(values, focal),
    )


def test_generic_is_not_target_specific():
    a = M.ownership_action(M.coefficients(M.toy_world("generic")))
    assert M.query_account(a)["varying_energy"] == pytest.approx(0)
    assert M.matched_contrast(a, M.TARGETS[2])["matched_minus_other"] == pytest.approx(
        0
    )
    assert np.linalg.norm(a) > 0


def test_selective_mechanism_has_correct_signed_contrast():
    a = M.ownership_action(M.coefficients(M.toy_world("selective")))
    assert M.matched_contrast(a, M.TARGETS[2])["matched_minus_other"] == pytest.approx(
        0.8
    )
    assert M.query_account(a)["varying_fraction"] == pytest.approx(5 / 6)


def test_contextual_mechanism_cancels_in_marginal_but_survives_conditionally():
    c = M.coefficients(M.toy_world("conditional"))
    assert np.allclose(M.ownership_action(c), 0)
    assert M.ownership_action(c, -1)[2] == pytest.approx(-0.8)
    assert M.ownership_action(c, 1)[2] == pytest.approx(0.8)


def test_owner_marker_without_content_does_not_pass_yg_gate():
    assert np.allclose(M.ownership_action(M.coefficients(M.toy_world("role_only"))), 0)


def test_null_directions_stay_null_and_shapes_are_not_flattened():
    assert M.cosine(np.zeros(6), np.ones(6)) is None
    with pytest.raises(ValueError):
        M.cosine(np.ones((6, 1)), np.ones(6))


def test_nonfinite_or_incomplete_observations_fail():
    with pytest.raises(ValueError):
        M.coefficients(np.ones((7, 6)))
    with pytest.raises(ValueError):
        M.query_account(np.array([1, 2, 3, 4, 5, np.nan]))
    with pytest.raises(ValueError):
        M.ownership_action(np.ones((8, 6)), 0)
