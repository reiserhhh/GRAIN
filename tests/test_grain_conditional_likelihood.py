#!/usr/bin/env python3
"""Tests of predictive units; synthetic checks are not mechanism evidence."""

import numpy as np
import pytest
from scipy.special import log_softmax, logsumexp

from grain_research.construction.math_conditional_likelihood import (
    paired_account,
    slice_account,
)


def account(logits, target):
    """Construct an independent full-vocabulary normalized example."""
    z = np.asarray(logits, float)
    return paired_account(
        z[:, :, target], logsumexp(z, -1), log_softmax(z, -1)[:, :, target], z.mean(-1)
    )


def test_alternatives_reverse_positive_observed_token_uplift():
    result = account([[[1, 5, 5]], [[0, 0, 0]]], 0)
    assert result["numerator"][0] > 0
    assert result["ratio"][0] < 0
    assert result["account_error"] < 1e-12


def test_independent_logit_gauges_cancel():
    z = np.arange(24, dtype=float).reshape(2, 3, 4) / 5
    a = account(z, 1)
    b = account(z + np.array([[7, -3, 9], [-2, 8, 1]])[:, :, None], 1)
    np.testing.assert_allclose(a["ratio"], b["ratio"], atol=1e-14)
    assert not np.allclose(a["numerator"], b["numerator"])
    np.testing.assert_allclose(
        b["centered_numerator"] - b["centered_normalizer"], b["ratio"], atol=1e-14
    )


def test_dual_rotation_preserves_logit_path():
    h = np.arange(18, dtype=float).reshape(2, 3, 3) / 8
    w = np.array([[1, 2, 3], [3, 0, -1], [1, -1, 1], [0, 1, 0]])
    q, _ = np.linalg.qr(np.arange(9).reshape(3, 3) + np.eye(3))
    a, b = account(h @ w.T, 0), account((h @ q) @ (w @ q).T, 0)
    np.testing.assert_allclose(a["ratio"], b["ratio"], atol=1e-14)


def test_chain_is_not_reset_slice_composition():
    result = slice_account([1, 2, 3, 4], [-3, -4], 2)
    assert result["full"] == 10
    assert result["reset_sum"] == -4
    assert result["carry_minus_reset"] == 14
    assert result["chain_error"] == 0


def test_short_paths_are_not_imputed():
    result = slice_account([1], None)
    assert result["full"] == 1 and result["reset_sum"] is None
    with pytest.raises(ValueError):
        slice_account([1], [0])


def test_nonfinite_or_misaligned_inputs_fail():
    with pytest.raises(ValueError):
        paired_account([[np.nan], [0]], [[0], [0]], [[0], [0]], [[0], [0]])
    with pytest.raises(ValueError):
        slice_account([1, 2, 3], [1, 2], 2)
