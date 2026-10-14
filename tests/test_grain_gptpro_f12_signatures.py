"""Functional invariance and finite-signature limitations."""

import numpy as np


def test_coordinate_and_action_transport():
    rng = np.random.default_rng(12)
    theta = rng.normal(size=(20, 2))
    x = np.column_stack([np.ones(4), [-1, -1 / 3, 1 / 3, 1]])
    s = np.array([[2.0, 0.4], [0.3, 0.5]])
    e = np.array([0.1, -0.1])
    changed = (theta + e) @ x.T - theta @ x.T
    transported = (theta @ s.T + e @ s.T) @ np.linalg.solve(s.T, x.T) - (
        theta @ s.T
    ) @ np.linalg.solve(s.T, x.T)
    np.testing.assert_allclose(changed, transported, atol=1e-12)


def test_four_signed_actions_have_two_dimensional_span():
    edits = np.array([[0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]])
    assert np.linalg.matrix_rank(edits) == 2


def test_grid_identity_does_not_imply_global_identity():
    grid = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    for z in grid:
        assert np.prod(z - grid) == 0
    assert np.isclose(0.2 * 0.1 * np.prod(-grid), 0.0022222222222222227)
