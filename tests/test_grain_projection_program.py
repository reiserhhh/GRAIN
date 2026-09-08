#!/usr/bin/env python3
"""Explicit kernel projection identity, order sensitivity and input validation."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.projection_program import (
    METHODS,
    pool_pairings,
    projection_from_embeddings,
    score_token_ids,
)


@pytest.mark.parametrize("method", list(METHODS))
def test_chart_identity_and_neural_free_token_replay(method):
    rng = np.random.default_rng(12)
    table = rng.normal(size=(20, 8)).astype(np.float32)
    r = table[[1, 2, 3]]
    chart = {
        "embeddings": r,
        "gradients": rng.normal(size=(2, 3, 8)),
        "projection_origin": np.array([0.2, 0.3]),
    }
    np.testing.assert_allclose(
        projection_from_embeddings(chart, r, method), [0.2, 0.3], atol=1e-6
    )
    value = score_token_ids(
        [1, 2, 3, 4, 5], [False, False, False, True, False], table, [chart], 0.1, method
    )
    assert value["score"] == pytest.approx(0.6, abs=1e-6)


def test_semantic_bag_invariance_and_order_condition():
    rng = np.random.default_rng(32)
    r, x, g = (
        rng.normal(size=(5, 8)),
        rng.normal(size=(7, 8)),
        rng.normal(size=(2, 5, 8)),
    )
    np.testing.assert_allclose(
        pool_pairings(r, x, g, "semantic"),
        pool_pairings(r, x[::-1], g, "semantic"),
        atol=2e-6,
    )
    assert not np.allclose(
        pool_pairings(r, x, g, "semantic_order"),
        pool_pairings(r, x[::-1], g, "semantic_order"),
    )


def test_reject_bad_program_inputs():
    with pytest.raises(ValueError):
        pool_pairings(np.ones((2, 3)), np.ones((3, 4)), np.ones((2, 2, 3)), "semantic")
    with pytest.raises(ValueError):
        score_token_ids([-1, 2], [True, False], np.ones((4, 2)), [], 0, "uniform")
