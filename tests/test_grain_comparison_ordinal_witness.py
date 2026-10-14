#!/usr/bin/env python3
"""Test independent cycle and acyclic witness certificates."""

from itertools import pairwise

import numpy as np

from scripts.verify_grain_comparison_observable_contract_hm228 import (
    topological_or_cycle,
)


def test_long_cycle_certificate():
    """Detect a directed ring even with no triangles."""
    a = np.zeros((6, 6), int)
    for i in range(6):
        a[i, (i + 1) % 6] = 1
    order, cycle = topological_or_cycle(a)
    assert not order
    assert cycle[0] == cycle[-1] and len(cycle) == 7
    assert all(a[i, j] == 1 for i, j in pairwise(cycle))


def test_acyclic_witness_and_isolated_vertices():
    """A topological order must include every vertex, not only selected edges."""
    a = np.zeros((7, 7), int)
    a[2, 0] = a[0, 5] = a[3, 5] = 1
    order, cycle = topological_or_cycle(a)
    assert not cycle and sorted(order) == list(range(7))
    pos = np.argsort(order)
    assert pos[2] < pos[0] < pos[5]
    assert pos[3] < pos[5]
