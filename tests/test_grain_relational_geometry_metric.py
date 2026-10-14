#!/usr/bin/env python3
"""Retained Gram blocks and exact generalized directional energy accounting."""

import numpy as np

from grain_research.construction import math_relational_geometry_metric as M


def test_all_four_blocks_reconstruct_not_delete_common_signal():
    rng = np.random.default_rng(126)
    x = rng.normal(size=(18, 6))
    g = x.T @ x
    blocks, share = M.decompose_gram(g)
    np.testing.assert_allclose(sum(blocks.values()), g, atol=1e-12)
    np.testing.assert_allclose(np.trace(blocks["common"]) / np.trace(g), share)
    np.testing.assert_allclose(np.trace(blocks["common_specific"]), 0, atol=1e-12)


def test_directional_gain_is_exact_norm_ratio():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(2, 30, 6))
    gx, gy = x.T @ x, y.T @ y
    result = M.metric_deformation(gx, gy)
    t = result["coefficient_directions"]
    ratios = np.sum((y @ t) ** 2, axis=0) / np.sum((x @ t) ** 2, axis=0)
    np.testing.assert_allclose(ratios, result["directional_energy_gains"], atol=1e-12)


def test_high_global_similarity_can_hide_specific_deformation():
    u = np.ones(6) / np.sqrt(6)
    gx = 1000 * np.outer(u, u) + np.eye(6)
    gy = gx + np.diag([0, 1, 2, 3, 4, 5])
    cosine = np.sum(gx * gy) / (np.linalg.norm(gx) * np.linalg.norm(gy))
    assert cosine > 0.9999
    assert max(M.metric_deformation(gx, gy)["directional_energy_gains"]) > 5
