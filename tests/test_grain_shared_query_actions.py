#!/usr/bin/env python3
"""Finite algebra and token-role invariants for the shared-query packet."""

import itertools

import numpy as np
import pytest

from grain_research.shared_query_actions import (
    cube_terms,
    exchange,
    make_edges,
    region_masks,
    worlds,
)


def test_full_factorial_interactions_and_allocations():
    """Known polynomial has exact main, pair and triple finite coefficients."""
    f = []
    for mask in range(8):
        c, p, q = [int(bool(mask & (1 << i))) for i in range(3)]
        f.append(
            10
            + 2 * c
            + 3 * p
            - 4 * q
            + 5 * c * p
            - 6 * p * q
            + 7 * c * q
            + 8 * c * p * q
        )
    r = cube_terms(f)
    for k, v in {
        "I_C": 2,
        "I_P": 3,
        "I_Q": -4,
        "I_CP": 5,
        "I_PQ": -6,
        "I_CQ": 7,
        "I_CPQ": 8,
    }.items():
        assert r[k] == pytest.approx(v)
    assert r["phi_C"] == pytest.approx(2 + 5 / 2 + 7 / 2 + 8 / 3)
    assert r["sum_error"] < 1e-12


def test_independent_order_allocation_identity():
    """Subset expansion equals marginal increments over all six orders."""
    rng = np.random.default_rng(701)
    for _ in range(10):
        f = rng.normal(size=8)
        result = cube_terms(f)
        expected = np.zeros(3)
        for order in itertools.permutations(range(3)):
            state = 0
            for i in order:
                new = state | (1 << i)
                expected[i] += (f[new] - f[state]) / 6
                state = new
        np.testing.assert_allclose(
            [result["phi_" + r] for r in "CPQ"], expected, atol=1e-12
        )


def test_exchange_partitions_and_preserves_sources():
    """Every token belongs to one carrier and no source array is modified."""
    b = np.arange(20, dtype=float).reshape(5, 4)
    d = b + 100
    p = [False, False, True, True, False]
    regions = region_masks(p)
    assert np.array_equal(sum(regions.values()), np.ones(5))
    for mask in range(8):
        m = exchange(b, d, p, mask)
        for i, r in enumerate("CPQ"):
            np.testing.assert_array_equal(
                m[regions[r]], (d if mask & (1 << i) else b)[regions[r]]
            )
    np.testing.assert_array_equal(b, np.arange(20).reshape(5, 4))


def test_edge_geometry_rejects_position_change():
    """Semantic donor labels cannot override incompatible token geometry."""
    design = []
    for a, d in itertools.product(range(2), range(2)):
        design.append(
            {
                "index": len(design),
                "frame": "f",
                "owner": "self",
                "query": "q",
                "arousal": a,
                "dejection": d,
                "ids": [0, 1, 2],
                "question_mask": [False, True, False],
            }
        )
    assert len(make_edges(design)) == 4
    design[1]["ids"].append(3)
    with pytest.raises(ValueError, match="position geometry"):
        make_edges(design)


def test_worlds_complete_and_independent_anchor_preserved():
    """No output-dependent exclusions or unrelated-anchor changes are allowed."""
    spec = {
        "frames": [{"id": "f", "arousal": ["a0", "a1"], "dejection": ["d0", "d1"]}],
        "owners": {"self": "s", "other": "o"},
        "unrelated_anchor": "constant",
    }
    result = worlds(spec)
    assert len(result) == 8 and len({r["id"] for r in result}) == 8
    assert all(r["body"].endswith("constant") for r in result)


@pytest.mark.parametrize("f", [[1] * 7, [1] * 9, [float("nan")] * 8])
def test_invalid_cube_rejected(f):
    """Incomplete or nonfinite exchanges must not produce a mechanism summary."""
    with pytest.raises(ValueError):
        cube_terms(f)
