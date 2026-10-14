#!/usr/bin/env python3
"""Synthetic graph/finite-account tests; real factorization checked in the runner."""

import numpy as np
import pytest

from grain_research.construction.math_event_context_routes import (
    ARMS,
    permission_mask,
    referenced_probe,
    route_account,
)


def fixture_row():
    """Two events, hidden padding, owner relay and two current tokens."""
    return {
        "input_ids": list(range(12)),
        "prefix": [0, 1],
        "events": [[2, 3], [5, 6]],
        "padding": [4, 7],
        "relay": [8, 9],
        "current": [10, 11],
    }


@pytest.mark.parametrize("c,r", ARMS)
def test_permissions(c, r):
    """Current access, causality and padding do not change with the intervention."""
    mask = permission_mask(fixture_row(), c, r)
    assert mask[6, 3] == bool(c)
    assert mask[9, 3] == bool(r)
    assert mask[11, 3] and mask[11, 6] and mask[11, 9]
    assert not mask[:, [4, 7]].any()
    assert not np.triu(mask, 1).any()


def test_recursive_independence():
    """No depth of allowed paths carries event1 to event2/relay in arm00."""
    graph = permission_mask(fixture_row(), 0, 0)
    reached = np.zeros(12, bool)
    reached[[2, 3]] = True
    for _ in range(17):
        reached |= (graph.astype(int) @ reached.astype(int)) > 0
    assert not reached[[5, 6, 8, 9]].any()
    assert reached[11]


def test_signed_decomposition():
    """Collector and signed route increments reconstruct the original full field."""
    values = np.random.default_rng(1).normal(size=(4, 8, 6))
    parts = route_account(values)
    np.testing.assert_allclose(parts.sum(0), values[3], atol=1e-14)
    np.testing.assert_allclose(parts[3], values[3] - values[1] - values[2] + values[0])


def test_reject_missing_partition_and_arm():
    """No unassigned padding or silent fractional permissions."""
    row = fixture_row()
    row["events"][0].append(5)
    with pytest.raises(ValueError):
        permission_mask(row, 0, 0)
    with pytest.raises(ValueError):
        permission_mask(fixture_row(), 0.5, 0)
    with pytest.raises(ValueError):
        route_account(np.zeros((3, 6)))


def test_probe_alignment():
    """References compare the same token slots, not pooled author summaries."""
    x = np.arange(24).reshape(12, 2, 1)
    probe = referenced_probe({"k": x, "v": x + 2}, [2, 6])
    np.testing.assert_array_equal(probe[0, :, 0, 0], [4, 12])
    assert probe.shape == (2, 2, 2, 1)
