#!/usr/bin/env python3
"""Finite controls test conservation, source independence and correct binding."""

import numpy as np
import pytest

from grain_research.construction import math_grain_mass_action as M


def source_fixture():
    """Synthetic four-dimensional marks with all declared source keys."""
    rng = np.random.default_rng(122)
    g = rng.normal(size=(288, 9, 4))
    mass = np.ones((288, 9))
    visible = np.tile(
        [False, True, True, True, True, True, True, True, False], (288, 1)
    )
    pairs = []
    for v, w, f, c, o in np.ndindex(3, 2, 6, 2, 2):
        low = len(pairs) * 2
        pairs.append(
            {"key": [v, w, f, 2 * c - 1, 2 * o - 1], "low": low, "high": low + 1}
        )
        g[low + 1] += np.array([1, 0.4, -0.3, 0.2]) * (f + 1)
    return g, mass, visible, pairs


def test_source_geometry_never_uses_held_marks_or_answers():
    g, m, mask, pairs = source_fixture()
    first = M.source_geometry(g, m, mask, pairs)
    g[96:] *= 1000
    second = M.source_geometry(g, m, mask, pairs)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_full_rank_geometry_is_affine_coordinate_equivariant():
    g, m, mask, pairs = source_fixture()
    geom = M.source_geometry(g, m, mask, pairs)
    transform = np.array(
        [[2, 0.3, 0, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 0.1], [0.1, 0, 0, 1]]
    )
    rotated = g @ transform.T + np.array([5, 4, 3, 2])
    new = M.source_geometry(rotated, m, mask, pairs)
    np.testing.assert_allclose(
        M.scores(g, m, mask, geom["axes"]),
        M.scores(rotated, m, mask, new["axes"]),
        atol=1e-11,
    )


def test_tilt_conserves_body_mass_and_keeps_exterior_and_zeros():
    m = np.array([[2, 1, 0, 1, 3]], float)
    mask = np.array([[False, True, True, True, False]])
    s = np.array([[100, -3, 2, 3, -100]], float)
    out = M.tilt(m, mask, s, 0.5)
    np.testing.assert_allclose(out.sum(1), m.sum(1), atol=1e-15)
    np.testing.assert_array_equal(out[:, [0, 2, 4]], m[:, [0, 2, 4]])
    assert (out[mask & (m > 0)] > 0).all()
    np.testing.assert_allclose(M.tilt(m, mask, s, 0), m, atol=0)


def test_wrong_positions_have_identical_mass_multiset_and_budget():
    g, m, mask, pairs = source_fixture()
    geom = M.source_geometry(g, m, mask, pairs)
    s = M.scores(g, m, mask, geom["axes"])
    arms = M.intervention_masses(m, mask, s, 0.5)
    for q, sign in np.ndindex(6, 2):
        a, b = arms[:, 1 + q * 4 + sign], arms[:, 3 + q * 4 + sign]
        for u in range(288):
            np.testing.assert_allclose(
                np.sort(a[u, mask[u]]), np.sort(b[u, mask[u]]), atol=1e-14
            )


def test_tilt_log_ratio_equals_score_difference():
    m = np.array([[0.2, 0.3, 0.5]])
    mask = np.ones_like(m, bool)
    s = np.array([[-1, 0.5, 2]])
    out = M.tilt(m, mask, s, 0.5)
    log_ratio = np.log(out / m)
    np.testing.assert_allclose(
        log_ratio[:, 1:] - log_ratio[:, :1], 0.5 * (s[:, 1:] - s[:, :1]), atol=1e-14
    )


def test_invalid_mass_and_empty_measure_are_rejected():
    for mass in (np.array([[-1.0, 2]]), np.zeros((1, 2))):
        with pytest.raises(ValueError):
            M.tilt(mass, np.ones((1, 2), bool), np.ones((1, 2)), 0.5)


def test_anonymous_four_coordinates_cannot_claim_six_independent_axes():
    g, m, mask, pairs = source_fixture()
    geom = M.source_geometry(g, m, mask, pairs)
    assert np.linalg.matrix_rank(geom["axes"]) <= 4
    assert geom["available"].all()


def test_response_field_cancels_assignment_even_and_focus_common_effects():
    _, _, _, pairs = source_fixture()
    r = np.ones((288, 25, 2, 2, 6))
    for p in pairs:
        for arm in range(25):
            r[p["high"], arm] += arm * 0.1
    result = M.response_field(r, pairs)
    np.testing.assert_allclose(result["actor_odd"], 0, atol=1e-14)
    assert len(M.selectivity(result["actor_odd"], np.ones(6))) == 288


def test_known_signed_target_response_survives_all_four_contrasts():
    """A true cue/behavior/actor/focus interaction must not cancel or swap axes."""
    _, _, _, pairs = source_fixture()
    r = np.zeros((288, 25, 2, 2, 6))
    for p in pairs:
        owner = p["key"][-1]
        f = p["key"][2]
        for side, polarity in (("low", -1), ("high", 1)):
            for q, alignment, sign_ix, focus, mode in np.ndindex(6, 2, 2, 2, 2):
                arm = 1 + q * 4 + alignment * 2 + sign_ix
                r[p[side], arm, focus, mode, f] = (
                    (2 * sign_ix - 1) * polarity * owner * (1 - 2 * focus) * (q + 1)
                )
    result = M.response_field(r, pairs)
    for v, w, f, c, q, alignment, mode in np.ndindex(3, 2, 6, 2, 6, 2, 2):
        expected = np.zeros(6)
        expected[f] = 2 * (q + 1)
        np.testing.assert_array_equal(
            result["actor_odd"][v, w, f, c, q, alignment, mode], expected
        )
    np.testing.assert_array_equal(result["actor_even"], 0)
