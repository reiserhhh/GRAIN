#!/usr/bin/env python3
"""Small correctness floor for the origin-preserving conditional operator."""

import numpy as np

from scripts.run_grain_conditional_response import fit_operator, grouped_folds, predict


def sample():
    rng = np.random.default_rng(53)
    groups = np.repeat(np.arange(36).astype(str), 8)
    z = np.repeat(rng.normal(size=(36, 4)), 8, axis=0)
    x = rng.normal(size=(len(z), 8))
    y = x[:, 0] * (1 + 2 * z[:, 0]) + 0.3 * x[:, 1]
    return x, z, y, groups


def test_folds_hold_all_author_rows_together():
    _, _, _, groups = sample()
    folds = grouped_folds(groups)
    assert set(folds) == {0, 1, 2}
    assert all(len(set(folds[groups == g])) == 1 for g in set(groups))


def test_conditional_model_recovers_known_gain_and_zero_origin():
    x, z, y, groups = sample()
    fit = np.arange(len(x)) < 24 * 8
    shared = fit_operator(x[fit], z[fit], y[fit], groups[fit], False, 1.0)
    conditional = fit_operator(x[fit], z[fit], y[fit], groups[fit], True, 1.0)
    p = predict(conditional, x[~fit], z[~fit])
    base = predict(shared, x[~fit], z[~fit])
    assert np.mean((p - y[~fit]) ** 2) < 0.05 * np.mean((base - y[~fit]) ** 2)
    assert np.allclose(predict(conditional, np.zeros_like(x), z), 0)


def test_frozen_models_ignore_test_batch_composition():
    x, z, y, groups = sample()
    model = fit_operator(x[:192], z[:192], y[:192], groups[:192], True, 10.0)
    one = predict(model, x[192:193], z[192:193])
    many = predict(model, x[192:], z[192:])
    assert np.allclose(one, many[:1])
    assert model["weights"].shape == (40,)
