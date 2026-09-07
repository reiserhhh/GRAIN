"""Readout centering and group splitting safeguards for cached operator analysis."""

import numpy as np
from sklearn.linear_model import Ridge

from scripts.analyze_grain_entity_operator import cross_predict, ridge_predict


def test_dual_ridge_matches_independent_sklearn_primal():
    rng = np.random.default_rng(42)
    x, y, test = (
        rng.normal(size=(6, 9)),
        rng.normal(size=(6, 3)),
        rng.normal(size=(2, 9)),
    )
    center = x.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((x - center) ** 2, axis=1)))
    model = Ridge(alpha=1).fit((x - center) / scale, y)
    assert np.allclose(
        ridge_predict(x, y, test), model.predict((test - center) / scale), atol=1e-12
    )


def test_held_setting_targets_cannot_change_their_own_prediction():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(8, 5)), rng.normal(size=(8, 4))
    groups, names = np.repeat(np.arange(4), 2), np.tile(["Alex", "Blair"], 4)
    for method in ("mean", "name_mean", "norm_ridge", "vector_ridge"):
        first = cross_predict(x, y, groups, names, method)
        changed = y.copy()
        changed[groups == 0] += 100
        second = cross_predict(x, changed, groups, names, method)
        assert np.allclose(first[groups == 0], second[groups == 0])


def test_constant_features_fall_back_to_training_mean():
    assert np.allclose(
        ridge_predict(np.ones((6, 9)), np.arange(12).reshape(6, 2), np.ones((2, 9))),
        [[5, 6], [5, 6]],
    )
