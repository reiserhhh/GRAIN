#!/usr/bin/env python3
"""Mechanism worlds for prospective evidence with and without its background."""

import numpy as np
import pytest

from grain_research.construction import math_evidence_background as M

S = np.array([1, -1, 1, 1, 1])


def test_partition():
    """Each selected event has exactly one declared role."""
    assert M.partition([0, 2, 4, 6], [1, 3, 5, 7]) == (8, 9, 10, 11)
    with pytest.raises(ValueError):
        M.partition([0, 1, 2, 3], [3, 4, 5, 6])
    with pytest.raises(ValueError):
        M.partition([0, 1, 2, 2], [3, 4, 5, 6])


def test_additive_evidence_null_interaction():
    """Adding the same background value cannot create a difference interaction."""
    y = np.arange(20, dtype=float).reshape(2, 2, 5)
    w = y[:, 0] - y[:, 1]
    a = M.account(w, y, S)
    np.testing.assert_allclose(a["raw"][:, 2], 0)


def test_target_background_not_common_only():
    """An A-conditioned background enters the target-specific account."""
    y = np.zeros((3, 2, 5))
    w = np.zeros((3, 5))
    w[:, 2] = 2
    a = M.account(w, y, S)
    np.testing.assert_allclose(a["specific"][:, 2, 2], 1.6)


def test_oriented_common_is_retained():
    """Evaluation-compatible movement is an account, not deleted evidence."""
    y = np.zeros((2, 2, 5))
    w = np.array([2, 3])[:, None] * S
    a = M.account(w, y, S)
    np.testing.assert_allclose(a["oriented_common"][:, 2], [2, 3])
    np.testing.assert_allclose(a["oriented_remainder"][:, 2], 0)
    np.testing.assert_allclose(
        a["raw"], a["oriented_common"][..., None] * S + a["oriented_remainder"]
    )


def test_negative_background_is_not_clipped():
    """Attenuation must not become a zero-filled failure or positive summary."""
    y = np.zeros((1, 2, 5))
    y[:, 0, 2] = 4
    a = M.account(np.zeros((1, 5)), y, S)
    assert a["raw"][0, 2, 2] == -4


def test_digit_gauge_and_logits():
    """High/low odds use all digits and ignore a shared logit gauge."""
    x = np.zeros((2, 2, 5, 10))
    x[..., 5:] = 1.2
    np.testing.assert_allclose(M.readings(x), 1.2)
    np.testing.assert_allclose(M.readings(x + 10), M.readings(x))
    with pytest.raises(ValueError):
        M.readings(x[:, :1])


def test_scope_validation():
    """Missing authors/targets cannot silently broadcast into a result."""
    with pytest.raises(ValueError):
        M.account(np.zeros((24, 5)), np.zeros((23, 2, 5)), S)
