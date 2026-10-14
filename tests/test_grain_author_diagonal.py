#!/usr/bin/env python3
"""Finite identity checks for source moments; no real data is required."""

import numpy as np
import pytest

from grain_research.construction import math_author_diagonal as M
from grain_research.construction.math_joint_target_atlas import construct


def fixture():
    """Make a complete paired source library with two authors in every profile."""
    rng = np.random.default_rng(239)
    return rng.normal(size=(64, 2, 5)), np.repeat(np.arange(32), 2)


def test_literal_and_atlas():
    x, c = fixture()
    k = x[:, 0] @ x[:, 1].T
    a, literal = M.accounts(k, c), M.literal_diagonal(k, c)
    for name in ("V", "D", "R", "U"):
        np.testing.assert_allclose(np.diag(a[name]), literal[name], atol=2e-14)
    f = construct(x, c)["fields"][0]
    np.testing.assert_allclose(a["V"], f[0] @ f[1].T, atol=2e-14)


def test_same_author_diagonal_is_character_neutral():
    x, c = fixture()
    d = np.diag(M.accounts(x[:, 0] @ x[:, 1].T, c)["D"])
    np.testing.assert_array_equal(d, np.repeat(d[0], 32))


def test_translation_is_not_a_psychological_diagonal_fraction():
    x, c = fixture()
    shifted = x + np.array([[4, 3, 2, 1, 0], [1, -2, 3, -4, 5]])
    a = M.accounts(x[:, 0] @ x[:, 1].T, c)
    b = M.accounts(shifted[:, 0] @ shifted[:, 1].T, c)
    for name in ("V", "U"):
        np.testing.assert_allclose(a[name][1:, 1:], b[name][1:, 1:], atol=2e-14)
    assert not np.isclose(a["D"][1, 1], b["D"][1, 1], rtol=0, atol=1e-12)


def test_repairing_preserves_atlas_not_identity_account():
    x, c = fixture()
    p = M.cyclic_pairing(c)
    assert not np.any(p == np.arange(len(c)))
    np.testing.assert_array_equal(c, c[p])
    k = x[:, 0] @ x[:, 1].T
    a, b = M.accounts(k, c), M.accounts(k[:, p], c)
    np.testing.assert_allclose(a["V"], b["V"], atol=2e-14)
    assert np.max(abs(a["U"] - b["U"])) > 0.001


def test_singleton_is_not_a_full_distinct_author_estimator():
    x, c = fixture()
    a = M.accounts((x[:, 0] @ x[:, 1].T)[1:, 1:], c[1:])
    assert a["U"] is None
    np.testing.assert_array_equal(a["singleton_profiles"], [0])


def test_missing_profile_rejected():
    x, c = fixture()
    with pytest.raises(ValueError, match="Missing profile"):
        M.accounts((x[:, 0] @ x[:, 1].T)[2:, 2:], c[2:])


def test_kernel_partition_at_two_origins():
    x, _ = fixture()
    kernels = M.kernel_spaces(x, np.eye(5)[:, :2])
    np.testing.assert_allclose(kernels[:, 0], kernels[:, 1] + kernels[:, 2])


def test_no_target_mean_paired_residual_world():
    # Balanced paired residuals cancel V; the distinct-index finite estimator
    # can still be negative and must not be clipped into a variance component.
    c = np.repeat(np.arange(32), 2)
    r = np.arange(1, 33, dtype=float) / 32
    x = np.stack((r, -r), axis=1).reshape(-1, 1)
    k = x @ x.T
    a = M.accounts(k, c)
    np.testing.assert_allclose(np.diag(a["V"])[1:], 0, atol=1e-14)
    assert np.all(np.diag(a["U"])[1:] < 0)


def test_distinct_index_correction_is_character_neutral():
    x, c = fixture()
    a = M.accounts(x[:, 0] @ x[:, 1].T, c)
    correction = np.diag(a["U"] - a["V"])
    np.testing.assert_allclose(correction, np.repeat(correction[0], 32), atol=2e-14)
