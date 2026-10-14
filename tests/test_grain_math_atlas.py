#!/usr/bin/env python3
"""Finite geometry checks with positive, curved and nonidentifiable examples."""

import numpy as np
import pytest

from grain_research.construction import math_atlas as A


def test_polynomial_chart_order():
    x = np.array([[2.0, 3.0]])
    np.testing.assert_array_equal(
        A.polynomial_coordinates(x, 2), [[2.0, 3.0, 4.0, 6.0, 9.0]]
    )
    with pytest.raises(ValueError):
        A.polynomial_coordinates(x, 3)


def test_affine_charts_close_without_noise():
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, (300, 2))
    charts = [
        x,
        x @ np.array([[1.2, 0.1], [-0.2, 0.7]]) + [0.3, 0.2],
        x @ np.array([[0.9, 0.3], [0.1, 1.1]]) - [0.4, 0.1],
    ]
    rows, _, _ = A.atlas_statistics(
        [c[:200] for c in charts], [c[200:] for c in charts], 1
    )
    assert max(r.get("cycle_defect", 0) for r in rows) < 1e-5


def test_curved_chart_needs_more_than_affine_map():
    src, app, _ = A.simulated_chart_world(71, "curved")
    rmse = []
    for degree in (1, 2):
        rows, _, _ = A.atlas_statistics(src, app, degree)
        rmse.append(next(r["observed_rmse"] for r in rows if r["path"] == "0_to_2"))
    assert rmse[1] < 0.1 * rmse[0]


def test_sign_collision_cannot_be_repaired_by_repeats():
    witness = A.collision_witness()
    assert witness["observation_difference"] == 0
    assert witness["signed_target_difference"] == 1.2
    src, app, truth = A.simulated_chart_world(71, "collision")
    model = A.fit_transition(src[2], src[0], 2)
    scores = A.transition(model, app[2])
    assert np.sqrt(np.mean((scores[:, 0] - truth["application"][:, 0]) ** 2)) > 0.4


def test_chart_cold_transform_uses_only_frozen_source():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    model = A.fit_transition(x, x + [2.0, -1.0], 1)
    first = A.transition(model, np.array([[0.4, 0.2]]))
    A.transition(model, np.full((20, 2), 1e9))
    np.testing.assert_array_equal(first, A.transition(model, np.array([[0.4, 0.2]])))
