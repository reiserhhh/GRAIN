#!/usr/bin/env python3
"""Source-only calibration and frozen representations have explicit limits."""

import numpy as np

from scripts.run_grain_gptpro_f11_grain_reader import fit_readout, predict_readout


def test_application_targets_cannot_change_fitted_readout():
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(12, 4)), rng.integers(0, 2, size=(12, 2)).astype(float)
    source = np.arange(8)
    first = fit_readout(x, y, source)
    y[8:] = np.nan
    assert fit_readout(x, y, source) == first
    raw, clipped = predict_readout(x, first)
    assert raw.shape == clipped.shape == (12, 2)
    np.testing.assert_array_equal(predict_readout(x[8:9], first)[0], raw[8:9])


def test_erased_content_difference_cannot_be_recovered_by_head():
    x = np.repeat(np.eye(3), 2, axis=0)
    y = np.column_stack([np.tile([0, 1], 3), np.repeat([0, 1, 1], 2)])
    b = fit_readout(x, y, np.arange(6))
    _, p = predict_readout(x, b)
    np.testing.assert_allclose(p[::2, 0], p[1::2, 0])
    np.testing.assert_allclose(p[:, 0], 0.5)
