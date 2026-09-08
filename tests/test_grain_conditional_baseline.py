#!/usr/bin/env python3
"""Conditional moments and leakage-free fixed author-reader witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_baseline import (
    attention_moments,
    author_folds,
    fixed_ridge,
    grouped_predictions,
)
from grain_research.vector_mechanisms.configuration_source import full_head


def test_moments_reproduce_full_grouped_query_heads():
    rng = np.random.default_rng(9)
    cache = {
        "q": rng.normal(size=(4, 6)),
        "k": rng.normal(size=(10, 2, 6)),
        "v": rng.normal(size=(10, 2, 6)),
    }
    mask = np.arange(10) < 6
    result = attention_moments(cache, mask)
    assert result["anchored_attention"] == pytest.approx(full_head(cache))
    assert result["anchored_moments"].shape == (24 * 2 + 8,)
    assert np.all((result["mass"] >= 0) & (result["mass"] <= 1))
    assert np.all(result["anchored_moments"][24:48] >= 0)


def test_folds_are_order_invariant_and_balanced():
    authors = [f"synthetic-{i}" for i in range(24)]
    a = author_folds(authors)
    b = author_folds(authors[::-1])[::-1]
    assert np.array_equal(a, b)
    assert np.array_equal(np.bincount(a), [6, 6, 6, 6])


def test_held_author_target_does_not_change_own_prediction():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(24, 8))
    y = x[:, 0] + rng.normal(size=24)
    f = author_folds([f"s-{i}" for i in range(24)])
    original, _ = grouped_predictions(x, y, f)
    y[f == 0] += 1000
    changed, _ = grouped_predictions(x, y, f)
    assert original[f == 0] == pytest.approx(changed[f == 0])


def test_normalization_uses_train_only_and_is_scale_invariant():
    rng = np.random.default_rng(11)
    x, z = rng.normal(size=(18, 7)), rng.normal(size=(6, 7))
    y = x[:, 0]
    first = fixed_ridge(x, y, z)
    assert first == pytest.approx(fixed_ridge(x * 4 + 3, y, z * 4 + 3))
    z[1:] *= 1e6
    assert first[0] == pytest.approx(fixed_ridge(x, y, z)[0])


def test_constant_features_fall_back_to_training_mean():
    assert fixed_ridge(
        np.ones((4, 3)), np.arange(4.0), np.ones((2, 3))
    ) == pytest.approx([1.5, 1.5])
