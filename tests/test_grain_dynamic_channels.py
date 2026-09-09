#!/usr/bin/env python3
"""Synthetic numerical checks independent of private materials."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.vector_mechanisms.dynamic_channels import (
    clustered_links,
    path_descriptors,
    residualize_descriptive,
    shapley_mse,
)


def test_permutation_roughness_expectation_and_reversal():
    x = np.random.default_rng(12).normal(size=(4, 8))
    expected = np.mean(
        [np.mean(np.diff(x[list(p)], axis=0) ** 2, 0) for p in permutations(range(4))],
        0,
    )
    np.testing.assert_allclose(expected, 2 * 4 / 3 * x.var(0))
    sd, d = path_descriptors(x)
    sd2, rev = path_descriptors(x[::-1])
    np.testing.assert_allclose(sd, sd2)
    np.testing.assert_allclose(d[:8], -rev[:8])
    np.testing.assert_allclose(d[8:16], rev[8:16])
    np.testing.assert_allclose(d[16:], -rev[16:])


def test_translation_invariance_and_constant():
    x = np.random.default_rng(4).normal(size=(4, 8))
    for a, b in zip(path_descriptors(x), path_descriptors(x + 8), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for value in path_descriptors(np.ones((4, 8))):
        np.testing.assert_allclose(value, 0)
    with pytest.raises(ValueError):
        path_descriptors(np.ones((3, 8)))


def test_author_clusters_preserve_motion_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 2, 2))
    y = x[:, :, 0] + rng.normal(size=(60, 2)) * 0.1
    r, lo, hi, p = clustered_links(x, y, draws=99)
    assert r[0] > 0.98 and lo[0] > 0.97 and hi[0] <= 1 and p[0] == 0.01
    assert abs(r[1]) < 0.2


def test_context_adjustment_is_descriptive_and_exact():
    z = np.random.default_rng(9).normal(size=(30, 2))
    y = z @ np.array([2.0, -3.0]) + 8
    np.testing.assert_allclose(residualize_descriptive(y, z), 0, atol=1e-12)


def test_within_comment_spans_are_lossless_and_not_days():
    from scripts.run_grain_within_comment_channels import lexical_slices

    text = "  " + "This is an example, with punctuation!\n" * 12
    slices = lexical_slices(text)
    assert len(slices) == 4 and "".join(slices) == text
    assert lexical_slices("short text") == []


def test_shapley_keeps_harmful_blocks_and_interactions():
    loss = {
        m: 10
        - (1 if m & 1 else 0)
        + (2 if m & 2 else 0)
        - (4 if m & 4 else 0)
        - (3 if m & 3 == 3 else 0)
        for m in range(8)
    }
    values = shapley_mse(loss)
    np.testing.assert_allclose(values, [2.5, -0.5, 4])
    assert values.sum() == loss[0] - loss[7]
    with pytest.raises(ValueError):
        shapley_mse({0: 1})
