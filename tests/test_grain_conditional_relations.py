#!/usr/bin/env python3
"""Known null and non-null worlds for conditional response relations."""

import numpy as np
import pytest

from grain_research.construction.math_conditional_relations import (
    relation_certificate,
    relation_projector,
    replicate_relation,
)


def test_common_linear_map_preserves_relations():
    """A shared operator must annihilate every source coefficient relation."""
    x = np.array([[1.0, 0], [0, 1], [1, 1], [2, 1]])
    b = np.array([[2.0, -1, 4], [1, 3, 2]])
    result, q, c, _ = relation_certificate(x, x @ b)
    assert result["source_kernel_dimension"] == 2
    assert result["relative_relation_defect"] < 1e-14
    assert np.linalg.norm(q @ x) < 1e-14
    assert c is not None


def test_broken_relation_not_rescued_by_operator_fit():
    """A target component in the source relation space cannot be transported."""
    x = np.array([[1.0, 0], [0, 1], [1, 1]])
    q, _, _, _ = relation_projector(x)
    y = x.copy() + q[:, :1] @ np.array([[5.0, -2]])
    result, _, _, _ = relation_certificate(x, y)
    assert result["relative_relation_defect"] > 0.1


def test_full_row_rank_allows_any_target():
    """Saturated finite inputs cannot test unrestricted linear transport."""
    x = np.array([[1.0, 0, 2, 0], [0, 1, 0, -1]])
    y = np.array([[8.0, -9, 5, 2], [-1, 7, 6, 4]])
    result, q, c, _ = relation_certificate(x, y)
    assert result["arbitrary_target_transport_possible"]
    assert result["arbitrary_invertible_extension_possible"]
    assert result["relative_relation_defect"] == 0
    assert np.count_nonzero(q) == 0
    assert c is None
    assert result["witness_target_rms"] is None


def test_lift_removes_a_real_finite_constraint():
    """Additional nonlinear coordinates can make a transport test vacuous."""
    x = np.array([[-1.0], [1]])
    y = np.array([[1.0], [2]])
    before, *_ = relation_certificate(x, y)
    after, *_ = relation_certificate(np.column_stack((x, x**2)), y)
    assert before["relative_relation_defect"] > 0
    assert after["arbitrary_target_transport_possible"]


def test_projector_invariant_under_invertible_source_coordinates():
    """Changing source coordinates does not change the coefficient relations."""
    x = np.array([[1.0, 0], [0, 1], [1, 1], [2, 1]])
    q, *_ = relation_projector(x)
    changed, *_ = relation_projector(x @ np.array([[2.0, 1], [0, 3]]))
    np.testing.assert_allclose(q, changed, atol=1e-14)


def test_null_and_signed_replication_accounts():
    """Held responses retain neutral instability and negative cross-products."""
    c = np.array([1.0, -1]) / np.sqrt(2)
    neutral = np.zeros((2, 2, 1))
    styled = np.array([[[1.0], [0]], [[-1.0], [0]]])
    result = replicate_relation(c, neutral, styled)
    assert result["neutral_rms"] == 0
    assert result["styled_signed_cross_product"] < 0
    assert result["change_signed_cross_product"] < 0


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.zeros(3), np.empty((0, 2))])
def test_invalid_observations_fail(bad):
    """Missing or malformed observations do not become zero measurements."""
    with pytest.raises(ValueError):
        relation_projector(bad)
