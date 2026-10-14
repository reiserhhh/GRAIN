"""Exact-design checks for F04 sample size and condition diversity."""

import numpy as np

from scripts.run_grain_gptpro_f04_information_recovery import information


def test_balanced_four_cell_information():
    z = np.tile([-1, -1 / 3, 1 / 3, 1], 6)
    matrix, covariance = information(z, 0.25)
    np.testing.assert_allclose(matrix, np.diag([96, 160 / 3]), atol=1e-12)
    np.testing.assert_allclose(covariance, np.diag([1 / 96, 3 / 160]), atol=1e-12)


def test_repetition_does_not_identify_absent_condition():
    for n in (8, 16, 32):
        matrix, covariance = information(np.full(n, -1), 0.25)
        assert np.linalg.matrix_rank(matrix) == 1
        assert covariance is None


def test_endpoints_more_informative_for_linear_slope():
    _, two = information(np.tile([-1, 1], 12), 0.25)
    _, four = information(np.tile([-1, -1 / 3, 1 / 3, 1], 6), 0.25)
    np.testing.assert_allclose(two[1, 1] / four[1, 1], 5 / 9)
