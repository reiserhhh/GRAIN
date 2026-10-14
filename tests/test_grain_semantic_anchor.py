#!/usr/bin/env python3
"""Numerical construction tests, not psychological-validity acceptance."""

import numpy as np
import pytest

from grain_research.construction import math_semantic_anchor as M


def sample():
    return np.random.default_rng(147).normal(size=(12, 4, 4))


def test_explicit_tensor_and_product_kernel():
    x = sample()
    phi = M.tensor_features(x)
    assert phi.shape == (12, 625)
    np.testing.assert_allclose(phi @ phi.T, M.product_kernel(x, x), atol=1e-10)


def test_sector_dimensions():
    sectors = M.feature_sectors()
    for mask in range(16):
        assert sum(sectors == mask) == 4 ** mask.bit_count()


def test_source_relink_preserves_all_conditional_marginals():
    x = sample()
    groups = [str(i // 3) for i in range(12)]
    seeds = [11, 23, 37] * 4
    r = M.relink_source(x, groups, seeds)
    assert not np.array_equal(r, x)
    for group in range(4):
        for v in range(4):
            assert sorted(map(tuple, r[group * 3 : group * 3 + 3, v])) == sorted(
                map(tuple, x[group * 3 : group * 3 + 3, v])
            )


def test_single_sectors_unchanged_joint_sectors_change():
    x = sample()
    r = M.relink_source(x, [str(i // 3) for i in range(12)], [11, 23, 37] * 4)
    weights = M.anchor_weights(
        ["E"] * 6 + ["N"] * 6, [1] * 3 + [-1] * 3 + [1] * 3 + [-1] * 3, ("E", "N")
    )
    coefficients, norms = M.compile_witness(x, weights, r)
    single = np.asarray([m.bit_count() <= 1 for m in M.feature_sectors()])
    np.testing.assert_allclose(
        coefficients[0, :, single], coefficients[2, :, single], atol=1e-14
    )
    assert np.linalg.norm(coefficients[0, :, ~single] - coefficients[2, :, ~single]) > 0
    assert np.all(norms > 0)


def test_cold_score_exact_signed_sector_sum():
    x = sample()
    origin, scale = M.source_units(x)
    z = M.standardize(x, origin, scale)
    a = np.zeros((12, 1))
    a[:6, 0], a[6:, 0] = 1 / 6, -1 / 6
    w, _ = M.compile_witness(z, a, z)
    score, parts = M.score_views(x, origin, scale, w)
    np.testing.assert_allclose(parts.sum(axis=-1), score, atol=1e-14)
    np.testing.assert_allclose(
        score[:, 0, 0], M.tensor_features(z) @ w[0, 0], atol=1e-14
    )


def test_independent_view_coordinate_rotations():
    x = sample()
    a = np.zeros((12, 1))
    a[:6, 0], a[6:, 0] = 1 / 6, -1 / 6
    rng = np.random.default_rng(148)
    q = np.stack([np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(4)])
    y = np.einsum("nvd,vde->nve", x, q)
    out = []
    for data in (x, y):
        origin, scale = M.source_units(data)
        z = M.standardize(data, origin, scale)
        w, _ = M.compile_witness(z, a, z)
        out.append(M.score_views(data, origin, scale, w)[0])
    np.testing.assert_allclose(*out, atol=1e-12)


@pytest.mark.parametrize("bad", [[], np.zeros((2, 4, 3)), np.full((2, 4, 4), np.nan)])
def test_illegal_observation_rejected(bad):
    with pytest.raises(ValueError):
        M.checked_views(bad)


def test_degenerate_source_view_rejected():
    with pytest.raises(ValueError):
        M.source_units(np.ones((3, 4, 4)))


def test_missing_source_pole_rejected():
    with pytest.raises(ValueError):
        M.anchor_weights(["E"] * 3, [1] * 3, ("E",))


def test_duplicate_seed_not_extra_joint_observation():
    with pytest.raises(ValueError):
        M.relink_source(np.ones((3, 4, 4)), ["one"] * 3, [11] * 3)
