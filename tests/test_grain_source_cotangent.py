#!/usr/bin/env python3
"""Source exclusion, exact attention JVP/VJP duality and inference input schema."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.source_cotangent import (
    QUERIES,
    predict_attention,
    source_means,
)


def test_source_mean_excludes_the_held_author_and_no_new_author_is_added():
    source = {
        (i, q): np.full((4, 4), i + 10 * j, float)
        for i in range(4)
        for j, q in enumerate(QUERIES)
    }
    model = source_means(source, QUERIES[0], held=3)
    np.testing.assert_allclose(model["own_g"], 1)
    np.testing.assert_allclose(model["pooled_g"], 6)
    assert model["source_indices"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(source_means(source, QUERIES[0])["own_g"], 1.5)
    with pytest.raises(ValueError):
        source_means(source | {(4, QUERIES[0]): np.ones((4, 4))}, QUERIES[0])


@pytest.mark.parametrize("kv", [1, 2])
def test_forward_direction_response_equals_pulled_back_field(kv):
    rng = np.random.default_rng(40 + kv)
    n, heads, d, width = 8, 4, 4, 7
    theta = np.tile(rng.normal(size=(n, d // 2)), (1, 2))
    inp = {
        "q": rng.normal(size=(heads, d)),
        "k": rng.normal(size=(n, kv, d)),
        "v": rng.normal(size=(n, kv, d)),
        "cos": np.cos(theta),
        "sin": np.sin(theta),
        "positions": np.arange(1, n - 1),
        "hidden": rng.normal(size=(n - 2, width)),
        "grain": rng.normal(size=(n - 2, 4)),
    }
    source = {(i, q): rng.normal(size=(heads, d)) for i in range(4) for q in QUERIES}
    weights = {k: rng.normal(size=(kv * d, width)) for k in ("k", "v")}
    model = source_means(source, QUERIES[0], held=0)
    result = predict_attention(inp, model, weights)
    assert result["response_key"].shape == (4, heads, d)
    scaled = model | {"own_g": model["own_g"] * 3}
    other = predict_attention(inp, scaled, weights)
    np.testing.assert_allclose(
        other["own_key_actions"], 3 * result["own_key_actions"], atol=1e-10
    )
    with pytest.raises(ValueError):
        predict_attention(inp | {"g": model["own_g"]}, model, weights)
