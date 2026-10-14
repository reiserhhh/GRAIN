#!/usr/bin/env python3
"""Analytic controls for finite qualified-action capacities, not personality gold."""

import numpy as np
import pytest

from grain_research.construction import math_qualified_channel as M


def test_inseparable_channel_bound():
    """The same vector cannot have unit own action and smaller foreign action."""
    result = M.capacity([[1.0]], [[1.0], [-1.0]])
    assert result["foreign_action_bound"] == pytest.approx(1)
    assert sum(result["lambda"]) == pytest.approx(1)


def test_separable_channel_bound():
    """Independent coordinates permit a foreign-invariant direct channel."""
    result = M.capacity([[1.0, 0.0]], [[0.0, 1.0]])
    assert result["foreign_action_bound"] == pytest.approx(0)
    assert np.dot([1, 0], result["w"]) >= 1 - M.TOL


def test_partial_separation():
    """Own opposition requires retaining a known amount of foreign response."""
    result = M.capacity([[1.0, 1.0], [1.0, -1.0]], [[2.0, 0.0]])
    assert result["foreign_action_bound"] == pytest.approx(2)


def test_infeasible_is_not_zero():
    """Impossible own orientation returns a certificate and no fake score."""
    result = M.capacity([[1.0], [-1.0]], [[1.0]])
    assert result["status"] == "NO_OWN_ORIENTATION"
    assert result["w"] is None


def test_role_products_preserve_context():
    """A role lift retains every original observation and uses no target truth."""
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = M.spaces(b, b, [-1, 1])
    assert np.array_equal(result["owner_joint49"][:, :2], b)
    assert np.array_equal(result["owner_joint49"][:, 3:], b * [[-1], [1]])
    scores = M.evaluate({"test": {"space": "marginal10", "w": [1, 0]}}, result)
    assert scores["direct_channels"]["test"] == [1, 3]
    assert scores["context_retained"] is result


def test_no_unknown_owner_imputation():
    """Unknown role cannot silently become a zero or assumed Person A."""
    with pytest.raises(ValueError):
        M.spaces([[1]], [[1]], [0])


def test_direct_nonlinear_role_control():
    """Known actor multiplication can split roles even with identical raw content."""
    raw = np.array([[-1], [1], [-1], [1]], float)
    roles = np.array([-1, -1, 1, 1])
    lifted = M.spaces(raw, raw, roles)["owner_joint49"]
    a, n = lifted[[3]] - lifted[[2]], lifted[[1]] - lifted[[0]]
    result = M.capacity(a, n)
    assert result["foreign_action_bound"] == pytest.approx(0)
    assert result["minimum_own_action"] >= 1 - M.TOL
