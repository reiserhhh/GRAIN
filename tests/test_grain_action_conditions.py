#!/usr/bin/env python3
"""Finite identities for the post-result condition diagnostic."""

import numpy as np

from scripts.diagnose_grain_action_geometry import grouped_covariance, split_functional


def test_functional_conditions_sum_to_symmetric_integral():
    """Origin/path/body terms reconstruct the actual declared two-path matrix."""
    a, b, c = np.random.default_rng(1).normal(size=(3, 4, 5))
    parts = split_functional(a, b, c)
    np.testing.assert_allclose(sum(parts.values()), (b + c) / 2, atol=1e-12)


def test_fold_covariance_does_not_invent_within_fold_signal():
    """A constant-per-fold control can correlate only through fold means."""
    groups = np.repeat(np.arange(4), 6)
    x = np.arange(24, dtype=float)
    y = -groups.astype(float)
    result = grouped_covariance(x, y, groups)
    assert abs(result["within_fold_covariance"]) < 1e-12
    assert result["closure_error"] < 1e-12
    assert abs(result["total_r"] - result["between_contribution_to_total_r"]) < 1e-12
