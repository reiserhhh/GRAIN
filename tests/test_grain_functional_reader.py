#!/usr/bin/env python3
"""Known-answer tests of estimand-aware fits and source/support isolation."""

import joblib
import numpy as np
import pytest

from grain_research.functional_reader import (
    ARMS,
    Block,
    FunctionalCoordinates,
    author_panel_permutation,
    block_loss,
    build_blocks,
    fit_functional,
    operator_profile,
    stacked_system,
)
from grain_research.response_accounting import difference_operator


def toy():
    """A misspecified shared slope: individual level and temporal slope differ."""
    a = np.linspace(-4, 4, 40)[:, None]
    t = np.array([-1.0, -1.0, 1.0, 1.0])[None, :]
    x = np.stack([np.ones((40, 4)), a + t], axis=-1)
    y = (a + 3 * t)[:, :, None]
    c = np.full((40, 4), "a")
    h = np.tile([0, 0, 1, 1], (40, 1))
    return x, y, c, h, np.ones(40, bool)


def test_function_loss_can_trade_level_for_response():
    b = build_blocks(*toy())
    point, _ = fit_functional(b, "point", 1e-4)
    within, _ = fit_functional(b, "within", 1e-4)
    assert block_loss(b["within"], within) < block_loss(b["within"], point)
    assert block_loss(b["point"], within) > block_loss(b["point"], point)
    assert within[1, 0] > point[1, 0]


@pytest.mark.parametrize("arm", list(ARMS))
def test_normal_equations_match_augmented_lstsq(arm):
    blocks = build_blocks(*toy())
    beta, audit = fit_functional(blocks, arm, 0.01)
    x, y = stacked_system(blocks, arm)
    penalty = np.eye(x.shape[1]) * 0.1
    penalty[0, 0] = 0
    expected = np.linalg.lstsq(
        np.vstack([x, penalty]),
        np.vstack([y, np.zeros((len(penalty), y.shape[1]))]),
        rcond=None,
    )[0]
    np.testing.assert_allclose(beta, expected, atol=1e-12)
    assert audit["stationarity_relative_max"] < 1e-12


def test_source_author_weights_and_intercept_annihilation():
    x, y, c, h, s = toy()
    x[0, 0], y[0, 0] = np.nan, np.nan
    blocks = build_blocks(x, y, c, h, s)
    b = blocks["point"]
    totals = [b.weights[b.authors == i].sum() for i in np.unique(b.authors)]
    np.testing.assert_allclose(totals, 1 / 40)
    for name in ("total", "within"):
        np.testing.assert_allclose(blocks[name].x[:, 0], 0, atol=1e-14)


def test_missing_shared_support_is_excluded_not_zero():
    x, y, c, h, s = toy()
    c[0] = ["a", "a", "b", "b"]
    s[1] = False
    x[2, 2:], y[2, 2:] = np.nan, np.nan
    b = build_blocks(x, y, c, h, s)
    assert len(np.unique(b["point"].authors)) == 40
    assert len(b["total"].authors) == 38
    assert len(b["within"].authors) == 37
    assert 0 not in b["within"].authors
    assert 2 not in b["total"].authors
    x[3, 0, 1] = np.nan
    with pytest.raises(ValueError, match="Partially"):
        build_blocks(x, y, c, h, s)


def test_energy_is_explicit_not_refit_on_validation():
    b = Block(
        np.ones((2, 1)), np.array([[1.0], [3.0]]), np.ones(2) / 2, np.array([0, 1]), 5.0
    )
    assert block_loss(b, np.zeros((1, 1)), energy=10) == 0.5
    assert block_loss(b, np.zeros((1, 1))) == 1


def test_whole_author_permutation_preserves_missingness():
    _, y, *_ = toy()
    y[2, 0] = np.nan
    p, order = author_panel_permutation(y)
    np.testing.assert_array_equal(p, y[order])
    assert np.isnan(p).sum() == 1
    np.testing.assert_array_equal(order, author_panel_permutation(y)[1])


def test_feature_transform_is_source_only_and_cold(tmp_path):
    rng = np.random.default_rng(12)
    g, z = rng.normal(size=(24, 3)), rng.normal(size=(24, 2))
    sub, ids = np.array(["a"] * 24), np.repeat(np.arange(12), 2)
    c = FunctionalCoordinates("joint_quadratic", 1.0).fit(g, z, sub, ids)
    mean_before, categories = c.g_scaler_.mean_.copy(), c.categories_.copy()
    held = c.design(g + 1000, z, np.array(["heldonly"] * 24))
    np.testing.assert_array_equal(c.g_scaler_.mean_, mean_before)
    assert c.categories_ == categories == ["a"]
    path = tmp_path / "coordinates.joblib"
    joblib.dump(c, path)
    np.testing.assert_array_equal(
        held, joblib.load(path).design(g + 1000, z, np.array(["heldonly"] * 24))
    )
    # Expanding first preserves within-comment variation.
    assert not np.allclose(
        c.design(g[:2], z[:2], sub[:2]).mean(0),
        c.design(g[:2].mean(0, keepdims=True), z[:2].mean(0, keepdims=True), sub[:1])[
            0
        ],
    )


def test_operator_profiles_keep_missing_conditions():
    _, y, c, h, strict = toy()
    c[0] = ["a", "a", "b", "b"]
    ops, common = [], []
    for i in range(len(y)):
        a, m = difference_operator(c[i], h[i], np.ones(4, bool))
        ops.append(a)
        common.append(m["within_measured"])
    p = operator_profile(y, np.stack(ops), np.asarray(common), strict)
    assert np.isnan(p[0, 2:4]).all()
    np.testing.assert_allclose(p[1:, 1, 0], 6)
    np.testing.assert_allclose(p[1:, 2, 0], 6)
    y[1, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        operator_profile(y, np.stack(ops), np.asarray(common), strict)


def test_reject_invalid_loss_and_alpha():
    b = build_blocks(*toy())
    with pytest.raises(ValueError):
        fit_functional(b, "point", 0)
    with pytest.raises(ValueError):
        stacked_system(b, "unknown")


def test_functional_rows_do_not_add_design_rank_or_information():
    blocks = build_blocks(*toy())
    point_x, point_y = stacked_system(blocks, "point")
    joint_x, _ = stacked_system(blocks, "balanced")
    assert np.linalg.matrix_rank(joint_x) == np.linalg.matrix_rank(point_x)
    # An exactly realizable teacher remains realizable under every functional.
    beta = np.array([[2.0], [-0.5]])
    x, _, c, h, strict = toy()
    exact = build_blocks(x, x @ beta, c, h, strict)
    for arm in ARMS:
        xx, yy = stacked_system(exact, arm)
        np.testing.assert_allclose(xx @ beta, yy, atol=1e-14)
    assert point_y.shape[0] == point_x.shape[0]
