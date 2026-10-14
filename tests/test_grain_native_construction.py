#!/usr/bin/env python3
"""Synthetic checks of identical-budget splits and source-frozen readouts."""

import numpy as np
import pytest

from grain_research.native_construction import (
    ARMS,
    NativeConstructionFeatures,
    fit_selected,
    fixed_splits,
    mean_r,
    paired_interval,
)


def fixture():
    """Generate varied five-criterion source data with four disjoint blocks."""
    rng = np.random.default_rng(14)
    blocks = {
        k: rng.normal(size=(55, w))
        for k, w in [("semantic", 8), ("grain", 4), ("affinity", 3), ("native", 5)]
    }
    y = blocks["native"] + rng.normal(size=(55, 5))
    return blocks, y, fixed_splits(np.arange(55) % 5)


def test_splits():
    _, _, split = fixture()
    assert not set(split["source"]) & set(split["held"])
    np.testing.assert_array_equal(
        np.sort(np.r_[split["train"], split["valid"]]), split["source"]
    )
    with pytest.raises(ValueError):
        fixed_splits([0, 1, 5])


@pytest.mark.parametrize("arm", ARMS[1:])
def test_affine_single_batch_and_held_labels(arm):
    blocks, y, split = fixture()
    obj = fit_selected(blocks, y, split, arm)
    feature, head = obj["features"], obj["head"]
    held = split["held"]
    expected = head.predict(feature.transform(blocks, held))
    intercept, weights = feature.affine(head, blocks)
    direct = intercept + sum(blocks[k][held] @ w for k, w in weights.items())
    np.testing.assert_allclose(expected, direct, atol=1e-12)
    single = np.vstack([head.predict(feature.transform(blocks, [i]))[0] for i in held])
    np.testing.assert_allclose(single, expected, atol=1e-12)
    changed_y = y.copy()
    changed_y[held] = 12345
    second = fit_selected(blocks, changed_y, split, arm)
    assert second["chosen"] == obj["chosen"]
    np.testing.assert_array_equal(second["head"].coef_, head.coef_)
    changed_x = {k: x.copy() for k, x in blocks.items()}
    for value in changed_x.values():
        value[held] = 98765
    transformed = NativeConstructionFeatures(arm).fit(changed_x, split["source"])
    np.testing.assert_array_equal(transformed.scaler_.mean_, feature.scaler_.mean_)


def test_correlation_and_pairing():
    _, y, _ = fixture()
    assert mean_r(y, y) == pytest.approx(1)
    assert mean_r(y, -y) == pytest.approx(-1)
    with pytest.raises(ValueError):
        mean_r(y, np.ones_like(y))
    interval = paired_interval(y, y, y, draws=30)
    assert interval["delta_mean_r"] == interval["lower"] == interval["upper"] == 0
    assert interval["invalid_draws"] == 0
