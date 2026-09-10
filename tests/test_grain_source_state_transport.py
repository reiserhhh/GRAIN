#!/usr/bin/env python3
"""Same-function derivatives, causal boundaries and source reference tests."""

import copy

import numpy as np
import pytest

from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES
from grain_research.vector_mechanisms.source_state_transport import (
    apply_map,
    body_directions,
    features,
    fit_maps,
    row_weights,
    source_set,
)


def materials():
    """Provide paired source states with a known causal update and unequal lengths."""
    rng = np.random.default_rng(52)
    items = []
    for i in range(4):
        for q in QUERIES:
            x = rng.normal(size=(9 + i, 6))
            y = x + 0.3 * np.cumsum(x, axis=0) / np.arange(1, len(x) + 1)[:, None]
            items.append({"index": i, "query": q, "x": x, "y": y})
    return items


@pytest.mark.parametrize("mode", ["pointwise", "causal_context"])
def test_same_function_finite_difference_and_lossless_factorization(mode):
    m = fit_maps(materials(), [0, 1, 2], rank=3)[mode]
    rng = np.random.default_rng(7)
    x = rng.normal(size=(11, 6))
    d = rng.normal(size=(2, 11, 6))
    y, dy, z, dz = apply_map(x, d, m)
    np.testing.assert_array_equal(y, x + z @ m["coef"])
    np.testing.assert_array_equal(dy, d + dz @ m["coef"])
    for f in range(2):
        eps = 1e-5
        plus = apply_map(x + eps * d[f], d, m)[0]
        minus = apply_map(x - eps * d[f], d, m)[0]
        np.testing.assert_allclose((plus - minus) / (2 * eps), dy[f], atol=1e-10)
    empty = m | {"coef": np.zeros_like(m["coef"])}
    np.testing.assert_array_equal(apply_map(x, d, empty)[0], x)
    np.testing.assert_array_equal(apply_map(x, d, empty)[1], d)


def test_no_future_input_and_body_to_query_transmission():
    models = fit_maps(materials(), [0, 1, 2, 3], rank=3)
    x = materials()[0]["x"]
    d = np.zeros((1, *x.shape))
    d[0, :3] = 1
    for name, m in models.items():
        y, dy, _, _ = apply_map(x, d, m)
        changed = x.copy()
        changed[5:] += 100
        np.testing.assert_array_equal(apply_map(changed, d, m)[0][:5], y[:5])
        if name == "pointwise":
            np.testing.assert_array_equal(dy[0, -1], 0)
        else:
            assert np.linalg.norm(dy[0, -1]) > 0


def test_source_exclusion_weighting_and_frozen_reference():
    items = materials()
    members = source_set("source4", 3)
    assert members == [0, 1, 2] and source_set("new8", 7) == [0, 1, 2, 3]
    models = fit_maps(items, members, rank=3)
    changed = copy.deepcopy(items)
    for item in changed:
        if item["index"] == 3:
            item["x"][:] = 1e6
            item["y"][:] = -1e6
    other = fit_maps(changed, members, rank=3)
    for name, m in models.items():
        for field in m:
            np.testing.assert_array_equal(m[field], other[name][field])
    w = row_weights(items)
    offset = 0
    for item in items:
        n = len(item["x"])
        assert abs(w[offset : offset + n].sum() - 1 / 8) < 1e-15
        offset += n
    mu = sum(item["x"].mean(0) / 8 for item in items)
    np.testing.assert_allclose(
        w @ np.concatenate([i["x"] for i in items]), mu, atol=1e-15
    )


def test_body_units_and_fixed_rank_failure():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    data = {"state": x, "positions": np.arange(2, 9), "grain": rng.normal(size=(7, 4))}
    d = body_directions(data)
    np.testing.assert_array_equal(d[:, -1], 0)
    np.testing.assert_array_equal(d[:, :2], 0)
    assert d.shape == (4, 12, 6) and np.isfinite(d).all()
    items = materials()
    for item in items:
        item["x"][:] = 1
    with pytest.raises(ValueError, match="rank"):
        fit_maps(items, [0, 1, 2], rank=3)


def test_layout_columns_have_zero_tangent():
    x = materials()[0]["x"]
    _, dphi = features(
        x, np.ones((2, *x.shape)), np.zeros(6), np.eye(6, 3), "causal_context"
    )
    np.testing.assert_array_equal(dphi[:, :, -3:], 0)
