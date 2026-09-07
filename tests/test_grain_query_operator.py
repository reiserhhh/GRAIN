"""Training-only normalization and fixed readout do not inspect held outcomes."""

import numpy as np

from scripts.analyze_grain_query_operator import fit_gap


def test_held_targets_cannot_change_fitted_gap_model():
    x = np.array([[1.0, 2.0], [-1.0, -2.0], [2.0, 1.0], [-2.0, -1.0]])
    y = np.array([3.0, -3.0, 4.0, -4.0])
    train = np.array([True, True, False, False])
    pred, beta, scale = fit_gap(x, y, train)
    y[~train] = 10000
    pred2, beta2, scale2 = fit_gap(x, y, train)
    np.testing.assert_array_equal(pred, pred2)
    np.testing.assert_array_equal(beta, beta2)
    assert scale == scale2
    assert pred[0] == -pred[1]
