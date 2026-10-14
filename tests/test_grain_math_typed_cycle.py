#!/usr/bin/env python3
"""Synthetic structural tests; no traits, private text or neural calls."""

import numpy as np
import pytest

from grain_research.construction import math_typed_cycle as M
from grain_research.construction.math_cycle_capacity import cycles
from scripts.verify_grain_typed_cycle_hm99 import alternate


def test_all_sectors_complete_and_orthonormal():
    b = M.sectors()
    assert [a.shape[-1] for a in b.values()] == [10, 1, 10, 50, 20, 30, 15]
    a = np.concatenate(list(b.values()), axis=-1)
    np.testing.assert_allclose(
        a.reshape(324, 136).T @ a.reshape(324, 136), np.eye(136), atol=1e-13
    )
    np.testing.assert_allclose(a.sum(1), 0, atol=1e-13)
    np.testing.assert_allclose(a + a.swapaxes(0, 1), 0, atol=1e-13)


def test_actual_parts_equal_conditional_average_and_parseval():
    rng = np.random.default_rng(990)
    b = M.sectors(3, 3)
    f = rng.normal(size=(9, 9, 4))
    _, parts = M.project(f, b)
    alt = alternate(f, 3, 3)
    for k in b:
        np.testing.assert_allclose(parts[k], alt[k], atol=1e-13)
    np.testing.assert_allclose(sum(parts.values()), cycles(f)[2], atol=1e-13)
    np.testing.assert_allclose(
        sum(np.sum(v * v) for v in parts.values()),
        np.sum(cycles(f)[2] ** 2),
        atol=1e-12,
    )


def test_pure_worlds_and_partial_swap():
    b = M.sectors(3, 3)
    for name, a in b.items():
        _, parts = M.project(a, b)
        for other, v in parts.items():
            np.testing.assert_allclose(v, a if other == name else 0, atol=1e-13)
    for name, sign in ((M.NAMES[5], -1), (M.NAMES[6], 1)):
        x = b[name].reshape(3, 3, 3, 3, -1)
        np.testing.assert_allclose(x.transpose(2, 1, 0, 3, 4), sign * x, atol=1e-13)


def test_relabeling_generators_commute_with_every_projector():
    rng = np.random.default_rng(991)
    b = M.sectors()
    f = rng.normal(size=(18, 18, 2))
    _, parts = M.project(f, b)
    for axis, n in ((0, 6), (1, 3)):
        for i in range(n - 1):
            grid = np.arange(18).reshape(6, 3)
            ix = np.arange(n)
            ix[i], ix[i + 1] = ix[i + 1], ix[i]
            p = (grid[ix, :] if axis == 0 else grid[:, ix]).ravel()
            _, relabeled = M.project(f[p][:, p], b)
            for name in b:
                np.testing.assert_allclose(
                    relabeled[name], parts[name][p][:, p], atol=1e-13
                )


def test_laws_positive_same_occupancy_and_traffic():
    d, p, m = M.positive_laws(M.sectors())
    assert p.min() > 0 and m.min() > 0
    np.testing.assert_allclose(p.sum((1, 2)), 1, atol=1e-13)
    np.testing.assert_allclose(p.sum(1), m.sum(1), atol=1e-13)
    np.testing.assert_allclose(
        p + p.transpose(0, 2, 1), m + m.transpose(0, 2, 1), atol=1e-13
    )
    assert len(d) == 136


def test_gradient_annihilation_and_undefined_cosine():
    g = np.arange(18, dtype=float)
    c, _ = M.project((g[:, None] - g[None, :])[..., None], M.sectors())
    for v in c.values():
        np.testing.assert_allclose(v, 0, atol=1e-13)
    assert np.isnan(M.compare(np.zeros((2, 1)), np.ones((2, 1)))["cosine"][0])


def test_invalid_product_support_rejected():
    with pytest.raises(ValueError):
        M.sectors(2, 3)
    with pytest.raises(ValueError):
        M.project(np.zeros((17, 17, 1)), M.sectors())
    with pytest.raises(ValueError):
        M.project(np.full((18, 18, 1), np.nan), M.sectors())
