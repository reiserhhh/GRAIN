#!/usr/bin/env python3
"""Small witnesses for measured interactions and conditional role references."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.comment_carrier import (
    joint_account,
    partition,
    reference_state,
    resample_role,
)


def test_additive_logits_can_have_score_interaction():
    """Output normalization alone can manufacture a nonzero probability quartet."""
    r = joint_account([0, 0], [0, 1], [0, 2], [0, 3], [0, 1])
    assert r["logit_interaction_rms"] == r["internal_score"] == 0
    assert abs(r["interaction"]) > 0.1
    assert r["interaction"] == pytest.approx(r["link_curvature"])


def test_mixed_logit_and_link_identity():
    """Common logit shifts leave both reported mechanisms unchanged."""
    z = np.array([[0, 0], [0, 1], [0, 2], [0, 4.0]])
    a = joint_account(*z, [0, 1])
    b = joint_account(*(z + np.arange(4)[:, None]), [0, 1])
    for key in a:
        assert a[key] == pytest.approx(b[key])
    assert a["interaction"] == pytest.approx(a["internal_score"] + a["link_curvature"])
    assert a["logit_interaction_rms"] > 0


def test_mixed_difference_is_integrated_cross_derivative():
    """For f(x,y)=x^2*y+3y, integration of 2x matches its exact quartet."""
    x, y = 2.0, 3.0
    f = lambda a, b: a * a * b + 3 * b
    mixed = f(x, y) - f(x, 0) - f(0, y) + f(0, 0)
    integral = y * x * (0 + 4 * x + 2 * x) / 6
    assert mixed == pytest.approx(integral)


def test_reference_uses_donors_not_target_values():
    """Target layout sets roles, but changing target content cannot change reference M/R."""
    g = [[2, 3], [5, 6]]
    p = np.array([False] * 8 + [True, True, False])
    h = np.arange(33, dtype=np.float32).reshape(11, 3)
    donors = [(h, g, p), (h + 4, g, p)]
    ref, changes = reference_state(h + 100, g, p, donors)
    other, _ = reference_state(h - 100, g, p, donors)
    _, gaps, body, rest = partition(g, p)
    assert changes == 0 and len(gaps) == 3
    np.testing.assert_array_equal(ref[:8], other[:8])
    np.testing.assert_array_equal(ref[8:], (h + 100)[8:])
    np.testing.assert_array_equal(ref[rest], (h + 2)[rest])
    for group in g:
        np.testing.assert_array_equal(
            ref[group], np.tile((h[group] + 2).mean(0), (2, 1))
        )
    assert not set(body) & set(rest)


def test_role_interpolation_and_invalid_partitions():
    """Unequal layouts have an explicit numerical rule, never silent donor truncation."""
    np.testing.assert_array_equal(
        resample_role(np.array([[0.0], [2.0]]), 3), [[0], [1], [2]]
    )
    with pytest.raises(ValueError):
        partition([[2, 3], [3, 4]], [False] * 6 + [True, False])
    with pytest.raises(ValueError):
        partition([[1, 3]], [False] * 6 + [True, False])


def test_partition_to_legacy_mean_projection_boundary():
    """Convert index arrays explicitly for the existing list-based projection API."""
    import torch
    from scripts.run_grain_mean_null_path import mean_null_direction

    h = torch.arange(27, dtype=torch.float32).reshape(9, 3)
    groups, _, body, rest = partition([[1, 2], [4, 5]], [False] * 7 + [True, False])
    direction = mean_null_direction(h, [x.tolist() for x in groups], [0, 1])
    for positions in groups:
        torch.testing.assert_close(
            (h + direction)[positions].mean(0), h[positions].mean(0)
        )
    torch.testing.assert_close(direction[rest], torch.zeros_like(direction[rest]))
    assert len(body) == 4
