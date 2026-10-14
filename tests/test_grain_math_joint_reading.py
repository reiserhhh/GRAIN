#!/usr/bin/env python3
"""Nulls and distinct mechanisms for complete role/content observation units."""

import numpy as np
import pytest

from grain_research.construction import math_joint_reading as M


def fields(seed=3):
    """Produce a positive, nontrivial toy attention law, not model evidence."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 1, (4, 5, 3, 2))
    p /= p.sum(axis=1, keepdims=True)
    return p, rng.normal(size=p.shape)


def test_complete_product_transform():
    p, c = fields()
    f = (p * c).sum(axis=(1, 3))
    terms = M.product_law(p, c)
    np.testing.assert_allclose(terms.sum(axis=1), M.coefficients(f), atol=1e-14)
    np.testing.assert_allclose(M.reconstruct(terms.sum(axis=1)), f, atol=1e-14)
    np.testing.assert_allclose(np.mean(f**2, axis=0), (M.coefficients(f) ** 2).sum(0))


def test_reader_created_joint_with_no_joint_value():
    s, o = M.BASIS[1], M.BASIS[2]
    p = np.stack([0.5 + 0.1 * s, 0.5 - 0.1 * s], axis=1)[:, :, None, None]
    c = np.stack([o, -o], axis=1)[:, :, None, None]
    terms = M.product_law(p, c)
    np.testing.assert_allclose(M.coefficients(c)[3], 0)
    np.testing.assert_allclose(terms[3, 0], 0)
    np.testing.assert_allclose(terms[3, 1], 0.2)
    np.testing.assert_allclose(4 * terms[3].sum(0), 0.8)


def test_carried_joint_and_uniform_additive_null():
    p = np.full((4, 2, 1, 1), 0.5)
    joint = np.broadcast_to(M.BASIS[3, :, None, None, None], p.shape)
    np.testing.assert_allclose(M.product_law(p, joint)[3, 0], 1)
    additive = np.broadcast_to(
        (2 + M.BASIS[1] + 3 * M.BASIS[2])[:, None, None, None], p.shape
    )
    np.testing.assert_allclose(M.product_law(p, additive)[3], 0)


def test_all_query_channels_and_permutation():
    p, c = fields()
    parts = M.query_product_law(p, c)
    np.testing.assert_allclose(parts.sum(0), M.product_law(p, c), atol=1e-14)
    order = [2, 0, 1]
    np.testing.assert_allclose(
        M.query_product_law(p[:, :, order], c[:, :, order]),
        parts[..., order],
        atol=1e-14,
    )


def test_shared_weights_can_have_target_specific_readings():
    p = np.full((4, 2, 2, 1), 0.5)
    c = np.broadcast_to(M.BASIS[3, :, None, None, None], p.shape).copy()
    c[:, :, 1] *= -2
    out = M.query_product_law(p, c)
    np.testing.assert_allclose(out[2:], 0)
    assert not np.isclose(out[1, 3, 0, 0], out[1, 3, 0, 1])


def test_positive_author_law_keeps_marginals_and_joint():
    p, c = fields()
    f = (p * c).sum(axis=(1, 3))
    a = M.product_law(p, c).sum(1)
    for law in ([0.1, 0.2, 0.3, 0.4], [0.375, 0.125, 0.125, 0.375]):
        np.testing.assert_allclose(M.author_score(a, law), np.asarray(law) @ f)
    plus, minus = (1 + 0.5 * M.BASIS[3]) / 4, (1 - 0.5 * M.BASIS[3]) / 4
    np.testing.assert_allclose(M.author_score(a, plus) - M.author_score(a, minus), a[3])


def test_primitives_gqa_direct_read():
    rng = np.random.default_rng(17)
    k, v = rng.normal(size=(2, 4, 3, 1, 5))
    q, d = rng.normal(size=(2, 3, 2, 5))
    p, c = M.primitives(k, v, q, d)
    expected = np.zeros((4, 3))
    for a in range(4):
        for target in range(3):
            for head in range(2):
                l = k[a, :, 0] @ q[target, head]
                w = np.exp(l - l.max())
                w /= w.sum()
                expected[a, target] += w @ (v[a, :, 0] @ d[target, head])
    np.testing.assert_allclose((p * c).sum(axis=(1, 3)), expected, atol=1e-14)


def test_invalid_inputs_fail_without_fabricated_scores():
    p, c = fields()
    with pytest.raises(ValueError):
        M.product_law(p * 2, c)
    with pytest.raises(ValueError):
        M.author_score(np.ones((4, 3)), [-0.1, 0.2, 0.3, 0.6])
    with pytest.raises(ValueError):
        M.coefficients(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        M.primitives(
            np.zeros((4, 0, 1, 2)),
            np.zeros((4, 0, 1, 2)),
            np.zeros((1, 1, 2)),
            np.zeros((1, 1, 2)),
        )
