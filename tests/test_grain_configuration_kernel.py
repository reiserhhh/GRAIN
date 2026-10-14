#!/usr/bin/env python3
"""Finite algebra tests for signed configuration measurement, not accuracy."""

import numpy as np
import pytest

from grain_research.construction import math_configuration_kernel as K


def test_explicit_tensor_equals_rectangular_kernel_contraction():
    rng = np.random.default_rng(4)
    x = [rng.normal(size=(11, 3)) for _ in range(3)]
    y = [rng.normal(size=(7, 3)) for _ in range(3)]
    tx = np.einsum("ni,nj,nk->ijk", *[a - a.mean(0) for a in x]) / 11
    ty = np.einsum("ni,nj,nk->ijk", *[a - a.mean(0) for a in y]) / 7
    assert K.interaction_inner(
        [a @ b.T for a, b in zip(x, y, strict=True)]
    ) == pytest.approx((tx * ty).sum())


def test_xor_has_zero_pairs_but_nonzero_triple():
    x = np.array([-1, -1, 1, 1])[:, None]
    y = np.array([-1, 1, -1, 1])[:, None]
    z = x * y
    grams = [a @ a.T for a in (x, y, z)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert K.interaction_inner([grams[i], grams[j]]) == pytest.approx(0)
    assert K.interaction_inner(grams) == pytest.approx(1)


def test_pair_only_dependence_not_called_lancaster_interaction():
    # A full product of a dependent XY pair with independent Z.
    x = np.repeat([-1, 1], 2)[:, None]
    z = np.tile([-1, 1], 2)[:, None]
    grams = [K.rbf(a, a, 1) for a in (x, x, z)]
    assert K.interaction_inner(grams) == pytest.approx(0, abs=1e-12)
    assert K.total_independence(grams) > 0


def test_weighted_tensor_equals_explicit_resampling():
    rng = np.random.default_rng(8)
    x, y = (
        [rng.normal(size=(5, 2)) for _ in range(3)],
        [rng.normal(size=(4, 2)) for _ in range(3)],
    )
    si, ai = np.array([0, 0, 2, 3, 3]), np.array([0, 0, 1, 3])
    w, v = np.bincount(si, minlength=5) / 5, np.bincount(ai, minlength=4) / 4
    cross = [a @ b.T for a, b in zip(x, y, strict=True)]
    repeated = [a[np.ix_(si, ai)] for a in cross]
    assert K.interaction_inner(cross, w, v) == pytest.approx(
        K.interaction_inner(repeated)
    )


def test_source_witness_does_not_use_application_batch_reference():
    rng = np.random.default_rng(3)
    x = [rng.normal(size=(9, 2)) for _ in range(3)]
    y = [rng.normal(size=(4, 2)) for _ in range(3)]
    ss = [a @ a.T for a in x]
    sa = [a @ b.T for a, b in zip(x, y, strict=True)]
    batch = K.source_witness(ss, sa)
    alone = K.source_witness(ss, [m[:, :1] for m in sa])
    assert alone[0] == pytest.approx(batch[0])


def test_coordinate_metric_transport_preserves_kernel():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(6, 3))
    a = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 2.0]])
    inv = np.linalg.inv(a)
    # Row coordinates x'=xA require precision'=A^-1 A^-T.
    assert np.allclose(K.rbf(x, y, 2), K.rbf(x @ a, y @ a, 2, inv @ inv.T))


def test_joint_author_renaming_changes_no_configuration():
    rng = np.random.default_rng(2)
    g = [rng.normal(size=(8, 5)) for _ in range(3)]
    s, a = rng.permutation(8), rng.permutation(5)
    assert K.interaction_inner(g) == pytest.approx(
        K.interaction_inner([m[np.ix_(s, a)] for m in g])
    )


def test_single_cut_leaves_other_pair_and_every_marginal_intact():
    rng = np.random.default_rng(7)
    g = [rng.normal(size=(9, 6)) for _ in range(3)]
    order = rng.permutation(6)
    result = K.cut_readings(g, np.array([order]))
    for j in range(3):
        shuffled = [m.copy() for m in g]
        shuffled[j] = shuffled[j][:, order]
        assert result[0, j] == pytest.approx(K.interaction_inner(shuffled))
        for k in set(range(3)) - {j}:
            assert np.array_equal(shuffled[k], g[k])
        assert np.allclose(np.sort(shuffled[j], axis=1), np.sort(g[j], axis=1))


def test_stratified_relink_retains_singleton_and_conditions():
    g = np.array([0, 0, 1, 1, 1, 2])
    order = K.permutation(6, np.random.default_rng(4), g)
    assert np.array_equal(np.sort(order), np.arange(6))
    assert np.array_equal(g[order], g) and order[-1] == 5


def test_balanced_views_are_complete_disjoint_and_equal_sized():
    venues = np.array(["A"] * 15 + ["B"] * 8 + ["C"] * 7 + ["D"] * 6)
    views = K.balanced_views(venues, np.arange(36)[::-1])
    assert views.shape == (3, 12)
    assert np.array_equal(np.sort(views.ravel()), np.arange(36))
    for view in views:
        assert np.all(np.diff(np.arange(36)[::-1][view]) >= 0)


def test_negative_signed_transport_is_not_clipped():
    x = np.array([-1, -1, 1, 1])[:, None]
    y = np.array([-1, 1, -1, 1])[:, None]
    z = x * y
    assert K.interaction_inner([x @ x.T, y @ y.T, z @ (-z).T]) == pytest.approx(-1)


@pytest.mark.parametrize("bad", [np.zeros((0, 2)), np.ones(3), np.array([[np.nan]])])
def test_invalid_observations_fail_explicitly(bad):
    with pytest.raises(ValueError):
        K.finite_matrix(bad)


def test_invalid_probability_or_permutation_not_rescued():
    with pytest.raises(ValueError):
        K.centered(np.eye(3), [0.2, 0.2, 0.2])
    with pytest.raises(ValueError):
        K.source_bandwidth(np.ones((3, 2)))
    with pytest.raises(ValueError):
        K.cut_readings([np.eye(3)] * 3, [[0, 0, 2]])
