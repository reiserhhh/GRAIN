#!/usr/bin/env python3
"""Post-account witness: functional losses can require non-target output directions."""

import numpy as np


def test_functional_optimum_can_leave_target_span():
    """A restricted optimum is not the full-space optimum of the same objective."""
    y, r = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    objective = lambda b: np.sum((b - y) ** 2) + (r @ (b - y)) ** 2 + np.sum(b * b)
    full = np.linalg.solve(2 * np.eye(2) + np.outer(r, r), y + r * (r @ y))
    restricted = np.array([2 / 3, 0])
    np.testing.assert_allclose(full, [5 / 8, 1 / 8], atol=1e-14)
    np.testing.assert_allclose(objective(full), 5 / 8, atol=1e-14)
    np.testing.assert_allclose(objective(restricted), 2 / 3, atol=1e-14)
    gradient = 2 * ((restricted - y) + r * (r @ (restricted - y)) + restricted)
    np.testing.assert_allclose(gradient, [0, -2 / 3], atol=1e-14)
    assert objective(full) < objective(restricted)
