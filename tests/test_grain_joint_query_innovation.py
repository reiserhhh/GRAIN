#!/usr/bin/env python3
"""Post-result algebra witnesses for the joint-observation interpretation."""

import numpy as np

from grain_research.vector_mechanisms import compiled_compatible_reader as previous
from grain_research.vector_mechanisms import joint_query_reader as joint


def test_joint_target_difference_is_the_previous_contrast_target():
    rng = np.random.default_rng(8910)
    current, native = rng.normal(size=(2, 12, 2, 5, 11))
    readers = rng.normal(size=(12, 2, 11))
    baseline = {"m": rng.normal(size=11), "d0": rng.normal(size=11)}
    units = np.arange(1.0, 6.0)
    a = joint.teaching(current, native, readers, baseline, units)
    b = previous.teaching(
        current, native, readers, baseline["m"], baseline["d0"], units
    )
    np.testing.assert_allclose(
        a["compatible_coordinates"][:, 0] - a["compatible_coordinates"][:, 1],
        b["compatible_coordinates"],
        atol=1e-12,
    )


def test_fixed_kernel_regression_commutes_with_mean_contrast_basis():
    rng = np.random.default_rng(8911)
    x, y = rng.normal(size=(12, 6)), rng.normal(size=(12, 10))
    a = x @ x.T + np.diag(np.linspace(0.1, 0.3, 12))
    identity = np.eye(5)
    transform = np.block([[identity / 2, identity], [identity / 2, -identity]])
    beta = np.linalg.solve(a, y)
    rotated = np.linalg.solve(a, y @ transform)
    np.testing.assert_allclose(rotated, beta @ transform, atol=1e-12)
    # Any common prediction row and its derivative obey the same identity.
    rows = rng.normal(size=(7, 12))
    np.testing.assert_allclose(rows @ rotated, (rows @ beta) @ transform, atol=1e-12)


def test_equal_query_shift_changes_pair_error_only_through_mean():
    rng = np.random.default_rng(8912)
    before = rng.normal(size=(8, 2, 5))
    after = before + rng.normal(size=(8, 1, 5))
    np.testing.assert_allclose(
        after[:, 0] - after[:, 1], before[:, 0] - before[:, 1], atol=1e-12
    )
    total_gain = np.sum(before**2 - after**2, axis=(0, 1))
    mean_gain = 2 * np.sum(before.mean(axis=1) ** 2 - after.mean(axis=1) ** 2, axis=0)
    np.testing.assert_allclose(total_gain, mean_gain, atol=1e-12)
