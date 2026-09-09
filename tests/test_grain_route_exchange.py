#!/usr/bin/env python3
"""Cross-query deltas must not silently align different question token rows."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.route_exchange import (
    endpoint_delta_patch,
    linear_cka,
    partition_routes,
)


def test_endpoint_only_transport_unequal_queries_and_identity():
    t = {"sites": {"attn:0": np.ones((7, 8))}}
    a = {"sites": {"attn:0": np.zeros((3, 8))}}
    b = {"sites": {"attn:0": np.full((3, 8), 2.0)}}
    c = [{"site": "attn:0", "head": 1}]
    p = endpoint_delta_patch(t, a, b, c, 4)["attn:0"]
    assert np.all(p[:-1] == 1) and np.all(p[-1, :4] == 1) and np.all(p[-1, 4:] == 3)
    assert np.all(t["sites"]["attn:0"] == 1)
    assert np.array_equal(
        endpoint_delta_patch(t, a, a, c, 4)["attn:0"], t["sites"]["attn:0"]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        endpoint_delta_patch(t, a, b, c + c, 4)


def test_partition_and_cka_have_distinct_meanings():
    a = [{"site": "attn:0", "head": 1}]
    b = a + [{"site": "mlp:2", "head": -1}]
    parts = partition_routes(a, b)
    assert parts["shared"] == a and parts["source_specific"] == b[1:]
    x = np.random.default_rng(12).normal(size=(6, 9))
    assert linear_cka(x, -x * 10 + 4) == pytest.approx(1)
    assert np.isnan(linear_cka(np.zeros((6, 2)), x))
