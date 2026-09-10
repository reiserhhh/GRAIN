#!/usr/bin/env python3
"""Verify both attention branches and a same-token contextual sign witness."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.attention_pullback import attention_pullback


@pytest.mark.parametrize("seed", range(4))
def test_three_pullbacks_match_finite_differences(seed):
    rng = np.random.default_rng(3800 + seed)
    q, k, v, g = (
        rng.normal(size=3),
        rng.normal(size=(5, 3)),
        rng.normal(size=(5, 4)),
        rng.normal(size=4),
    )
    result = attention_pullback(q, k, v, g)
    eps = 1e-5
    for target, expected in (
        (q, result["query"]),
        (k, result["keys"]),
        (v, result["values"]),
    ):
        delta = rng.normal(size=target.shape)
        original = target.copy()
        target[:] = original + eps * delta
        positive = attention_pullback(q, k, v, g)["output"] @ g
        target[:] = original - eps * delta
        negative = attention_pullback(q, k, v, g)["output"] @ g
        target[:] = original
        assert (positive - negative) / (2 * eps) == pytest.approx(
            float(np.sum(expected * delta)), abs=2e-9
        )
    assert abs(result["score_gradient"].sum()) < 1e-14


def test_same_local_token_query_and_attention_can_change_response_sign():
    # h0=(0,1), Wk=(1,0), Wv=(0,1); only the other token's value changes.
    a = attention_pullback([1], [[0], [0]], [[1], [0]], [1])
    b = attention_pullback([1], [[0], [0]], [[1], [3]], [1])
    np.testing.assert_array_equal(a["attention"], b["attention"])
    assert a["keys"][0, 0] == pytest.approx(0.25)
    assert b["keys"][0, 0] == pytest.approx(-0.5)
    assert a["values"][0, 0] == b["values"][0, 0] == 0.5


def test_equal_values_remove_selection_but_not_value_transport():
    r = attention_pullback([1, 2], [[1, 0], [3, 1]], [[4], [4]], [2])
    np.testing.assert_allclose(r["keys"], 0, atol=1e-14)
    assert np.linalg.norm(r["values"]) > 0
    with pytest.raises(ValueError):
        attention_pullback([1], [[0, 1]], [[1]], [1])
