#!/usr/bin/env python3
"""Reference, identity and conditional-support checks on synthetic data."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.paired_support import fit_support, predict_support


def fixture():
    """Create an even author offset plus an odd order response."""
    rng = np.random.default_rng(70)
    a, b = rng.normal(size=(24, 3)), rng.normal(size=(24, 3))
    xs = [np.stack([a, a], 1), np.stack([b, -b], 1)]
    y = xs[0][..., :1] + 0.4 * xs[1][..., 1:2]
    return xs, y


def test_original_moments_identical_across_support():
    xs, y = fixture()
    models = [fit_support(xs, y, np.arange(18), paired=v) for v in (False, True)]
    for a, b in zip(models[0]["centers"], models[1]["centers"], strict=True):
        np.testing.assert_array_equal(a, b)
    assert models[0]["scales"] == models[1]["scales"]
    assert models[1]["head"].alpha == 2 * models[0]["head"].alpha


def test_held_changes_do_not_change_fitted_function():
    xs, y = fixture()
    first = fit_support(xs, y, np.arange(18), True)
    altered = [x.copy() for x in xs]
    for x in altered:
        x[18:] += 1234
    changed_y = y.copy()
    changed_y[18:] += 300
    second = fit_support(altered, changed_y, np.arange(18), True)
    np.testing.assert_allclose(predict_support(first, xs), predict_support(second, xs))
    np.testing.assert_allclose(
        predict_support(first, [x[18:] for x in xs]), predict_support(first, xs)[18:]
    )


def test_known_even_odd_function_transports():
    xs, y = fixture()
    model = fit_support(xs, y, np.arange(18), True, penalty=1e-8)
    np.testing.assert_allclose(predict_support(model, xs)[18:], y[18:], atol=2e-6)


def test_invalid_group_rejected():
    xs, y = fixture()
    with pytest.raises(ValueError):
        fit_support(xs, y, [0, 0, 1], True)


def test_general_paired_loss_even_odd_account():
    rng = np.random.default_rng(114)
    actual, predicted = rng.normal(size=(2, 31, 2))
    error = actual - predicted
    even = error.mean(1)
    odd = (error[:, 0] - error[:, 1]) / 2
    np.testing.assert_allclose(np.mean(error**2), np.mean(even**2 + odd**2))


def test_orbit_orthogonality_does_not_erase_natural_orientation_link():
    y = np.linspace(-2, 3, 21)
    odd = np.stack([y, -y], axis=1)
    repeated_y = np.repeat(y, 2)
    assert np.corrcoef(y, odd[:, 0])[0, 1] == pytest.approx(1)
    assert np.cov(odd.ravel(), repeated_y, ddof=0)[0, 1] == pytest.approx(0)
