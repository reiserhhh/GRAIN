#!/usr/bin/env python3
"""Author recurrence can succeed, fail or tie under the same coupling rule."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_author_coupling as M


def bits():
    return np.array(list(itertools.product((-1.0, 1.0), repeat=4)))


def toy_world(mode, authors=40):
    rng = np.random.default_rng(24)
    g = bits()
    panels = []
    for _ in range(authors):
        shared = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        for _ in range(3):
            r = (
                np.linalg.qr(rng.normal(size=(4, 4)))[0]
                if mode == "independent"
                else shared
            )
            if mode == "constant":
                r = np.eye(4)
            order = rng.permutation(len(g))
            panels.append(M.panel_statistics(g[order], (g @ r)[order]))
    return M.stack_statistics(panels)


def test_identical_locations_and_repeatable_relations_are_distinguishable():
    stats = toy_world("stable")
    ref = M.fit_reference({k: v[:60] for k, v in stats.items()})
    p = M.profiles({k: v[60:] for k, v in stats.items()}, ref)
    d = M.squared_distances(p["coupling"][::3], p["coupling"][1::3])
    loc = M.squared_distances(p["joint_location"][::3], p["joint_location"][1::3])
    assert M.candidate_wins(d, loc)[0].diagonal().mean() == 1
    assert M.candidate_wins(loc, loc)[0].diagonal().mean() == 0.5


def test_constant_shared_relation_does_not_create_author_evidence():
    stats = toy_world("constant")
    ref = M.fit_reference(stats)
    assert ref["neutral_block"][2]
    p = M.profiles(stats, ref)["coupling"]
    d = M.squared_distances(p[::3], p[1::3])
    assert M.candidate_wins(d, d)[0].diagonal().mean() == 0.5


def test_independent_relations_do_not_pass_as_repeatability():
    stats = toy_world("independent", 80)
    ref = M.fit_reference({k: v[:120] for k, v in stats.items()})
    p = M.profiles({k: v[120:] for k, v in stats.items()}, ref)
    d = M.squared_distances(p["coupling"][::3], p["coupling"][1::3])
    loc = M.squared_distances(p["joint_location"][::3], p["joint_location"][1::3])
    assert 0.35 < M.candidate_wins(d, loc)[0].diagonal().mean() < 0.65


def test_shift_changes_pairing_but_preserves_marginals():
    rng = np.random.default_rng(5)
    g = rng.normal(size=(13, 4))
    y = g @ rng.normal(size=(4, 5))
    s = M.panel_statistics(g, y)
    shifted = np.roll(y, len(y) // 2, axis=0)
    np.testing.assert_allclose(y.mean(0), shifted.mean(0), atol=1e-14)
    np.testing.assert_allclose(y.T @ y, shifted.T @ shifted, atol=1e-12)
    assert np.linalg.norm(s["cross"] - s["shifted"]) > 0.1


def test_tensor_metric_invertible_coordinate_invariance():
    rng = np.random.default_rng(81)
    x, y = rng.normal(size=(4, 4)), rng.normal(size=(5, 5))
    sg, sy = x @ x.T + np.eye(4), y @ y.T + np.eye(5)
    e = rng.normal(size=(4, 5))
    a, b = (
        rng.normal(size=(4, 4)) + 3 * np.eye(4),
        rng.normal(size=(5, 5)) + 3 * np.eye(5),
    )
    original = M.trace_distance(e, sg, sy)
    moved = M.trace_distance(a @ e @ b.T, a @ sg @ a.T, b @ sy @ b.T)
    wg, _, _ = M.inverse_root(sg)
    wy, _, _ = M.inverse_root(sy)
    assert moved == pytest.approx(original, rel=1e-10)
    assert np.sum((wg @ e @ wy) ** 2) == pytest.approx(original, rel=1e-12)


def test_source_units_do_not_depend_on_held_profiles():
    stats = toy_world("stable")
    source = {k: v[:60].copy() for k, v in stats.items()}
    held = {k: v[60:].copy() for k, v in stats.items()}
    ref = M.fit_reference(source)
    before = {k: v.copy() for k, v in ref.items()}
    held["cross"] *= 100
    M.profiles(held, ref)
    for key, values in before.items():
        np.testing.assert_array_equal(values, ref[key])


def test_singular_reference_fails_without_rank_or_ridge_rescue():
    with pytest.raises(ValueError, match="Degenerate"):
        M.inverse_root(np.ones((4, 4)))


def test_candidate_null_all_strangers_has_exact_half_mean():
    rng = np.random.default_rng(64)
    d = rng.uniform(size=(8, 8))
    all_win, _ = M.candidate_wins(d, d)
    np.testing.assert_allclose(all_win.mean(1), 0.5, atol=1e-14)


def test_hard_strangers_are_location_selected_not_coupling_selected():
    location = np.tile(np.arange(6, dtype=float), (6, 1))
    coupling = location[:, ::-1].copy()
    _, hard = M.candidate_wins(coupling, location)
    # For nominated identity 5, the three nearest other location profiles are 0,1,2.
    assert hard[0, 5] == 1
    # A uniform nominated identity need not have a .5 hard-stranger null.
    _, location_hard = M.candidate_wins(location, location)
    assert location_hard.mean() != pytest.approx(0.5)


def test_bootstrap_resamples_author_endpoints_jointly():
    values = np.column_stack((np.arange(10) / 10, np.arange(10) / 10 + 0.1))
    means, ci, bootstrap = M.bootstrap_summary(values, repetitions=200)
    np.testing.assert_allclose(bootstrap[:, 1] - bootstrap[:, 0], 0.1, atol=1e-14)
    assert means[0] == pytest.approx(0.45)
    assert ci.shape == (2, 2)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_incomplete_stranger_support_fails(n):
    with pytest.raises(ValueError):
        M.candidate_wins(np.zeros((n, n)), np.zeros((n, n)))
