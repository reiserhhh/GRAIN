#!/usr/bin/env python3
"""Finite information witnesses must distinguish loss from nonlinearity."""

import numpy as np
import pytest

from grain_research.construction import math_observation_fibers as M


def test_exact_not_nearby_and_signed_zero():
    g = np.array([[0, -0.0], [-0.0, 0], [1e-16, 0]])
    _, inverse, counts = M.exact_partition(g)
    assert inverse[0] == inverse[1] != inverse[2]
    assert sorted(counts) == [1, 2]


def test_no_repeated_inputs_proves_only_finite_lookup():
    g, y = np.arange(4)[:, None], np.array([[9], [-2], [30], [1]])
    p = M.empirical_projection(g, y, np.full(4, 0.25))
    assert p["risk"].item() == 0
    assert not M.fiber_extrema(g, y)
    _, seen = M.lookup(p, [[4]])
    assert not seen.item()


def test_weighted_projection_and_pythagorean_identity():
    g = np.array([[0], [0], [1], [1]])
    y = np.array([[1, 2], [3, 4], [7, 1], [9, 3]], float)
    w = np.array([0.1, 0.3, 0.2, 0.4])
    p = M.empirical_projection(g, y, w)
    np.testing.assert_allclose(p["means"], [[2.5, 3.5], [25 / 3, 7 / 3]])
    candidate = p["means"] + 0.25
    lhs = np.sum(w[:, None] * (y - candidate[p["inverse"]]) ** 2, axis=0)
    np.testing.assert_allclose(lhs, p["risk"] + 0.25**2)


def test_author_weights_do_not_reward_more_comments():
    w = M.author_weights([0, 0, 0, 1])
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2])


def test_unseen_exact_inputs_are_not_zero_scores():
    p = M.empirical_projection([[0], [0]], [[1], [3]], [0.5, 0.5])
    y, seen = M.lookup(p, [[0], [1e-16]])
    assert seen.tolist() == [True, False]
    assert y[0, 0] == 2 and np.isnan(y[1, 0])


def test_two_point_equal_input_measure_witness():
    g, y = np.array([[2, 3], [2, 3]]), np.array([[-1], [3]])
    row = M.fiber_extrema(g, y)[0]
    assert row["certified"] and row["worst_absolute_error_bound"] == 2
    plus, minus = np.array([0.25, 0.75]), np.array([0.75, 0.25])
    for power in range(1, 10):
        np.testing.assert_array_equal(plus @ g**power, minus @ g**power)
    assert ((plus - minus) @ y).item() == row["two_distribution_mean_gap"]


@pytest.mark.parametrize("world", M.WORLDS)
def test_world_mechanism_not_fake_random_scores(world):
    material = M.synthetic_world(world, 20261009)
    p = M.empirical_projection(
        material["g"], material["y"], M.author_weights(material["author"])
    )
    e = M.roundoff_envelope(material["h"], material["W"])
    rows = M.fiber_extrema(material["g"], material["y"], e)
    if world == "omitted_coordinate":
        assert min(p["risk"]) > 0.1 and all(row["certified"] for row in rows)
    else:
        assert max(p["risk"]) < 1e-24 and not any(row["certified"] for row in rows)
    source = material["source"]
    ref = M.empirical_projection(
        material["g"][source],
        material["y"][source],
        M.author_weights(material["author"][source]),
    )
    _, seen = M.lookup(ref, material["g"][~source])
    assert seen.all() if world != "injective_support" else not seen.any()


def test_roundoff_not_fuzzy_equality_or_personality_threshold():
    h, w = np.eye(2), np.ones((2, 1))
    e = M.roundoff_envelope(h, w)
    rows = M.fiber_extrema(np.zeros((2, 1)), np.array([[1], [1 + 1e-15]]), e)
    assert rows[0]["raw_nonconstant"] and not rows[0]["certified"]


def test_invalid_weights_and_nonfinite_inputs_fail():
    with pytest.raises(ValueError, match="finite"):
        M.exact_partition([[np.nan]])
    with pytest.raises(ValueError, match="positive"):
        M.empirical_projection([[0], [0]], [[1], [2]], [1, 0])


def test_held_mutation_cannot_change_source_projection():
    a = M.synthetic_world("omitted_coordinate", 1)
    source = a["source"]
    p = M.empirical_projection(
        a["g"][source], a["y"][source], M.author_weights(a["author"][source])
    )
    before = p["means"].copy()
    a["y"][~source] *= 100
    M.lookup(p, a["g"][~source])
    np.testing.assert_array_equal(before, p["means"])
