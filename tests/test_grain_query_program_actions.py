#!/usr/bin/env python3
"""Synthetic checks for preserving exact readers while ablating carriers."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.affine_program import compile_affine, score_affine
from grain_research.vector_mechanisms.context_program import score_context_charts
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.projection_program import pool_pairings
from grain_research.vector_mechanisms.query_program_actions import (
    chart_carriers,
    context_scores,
)
from grain_research.vector_mechanisms.source_span import fit_linear, predict_linear
from grain_research.vector_mechanisms.state_transport import unpack_state


def example(seed=1):
    """Create two source charts with equal question coordinates and new context rows."""
    rng = np.random.default_rng(seed)
    mask = np.array([0, 0, 0, 0, 1, 1, 0], bool)
    charts = [
        {
            "hidden": rng.normal(size=(6, 4)).astype(np.float32),
            "gradients": rng.normal(size=(2, 6, 4)).astype(np.float32),
            "mask": np.array([0, 0, 0, 1, 1, 0], bool),
            "projection_origin": rng.normal(size=2),
        }
        for _ in range(2)
    ]
    return charts, rng.normal(size=(7, 4)).astype(np.float32), mask


def test_carrier_sum_matches_existing_program_and_does_not_mutate():
    charts, h, mask = example()
    before = h.copy()
    result = context_scores(charts, h, mask, 0.2)
    expected = score_context_charts(charts, h, mask, 0.2)["mean"]["score"]
    assert result["context_all"] == pytest.approx(expected, abs=2e-6)
    assert result["context_all"] == pytest.approx(
        result["context_C"] + result["context_PQ"] - result["context_origin"]
    )
    np.testing.assert_array_equal(h, before)


def test_query_change_leaves_context_carrier_unchanged():
    charts, h, mask = example()
    altered = h.copy()
    altered[mask] += 2
    a, b = [context_scores(charts, v, mask, 0.2) for v in (h, altered)]
    assert a["B_C"] == b["B_C"]
    assert a["B_Q"] == b["B_Q"]
    assert a["B_P"] != b["B_P"]


def test_origin_is_exact_and_bad_coordinates_fail():
    charts, _, _ = example()
    for c in charts:
        result = chart_carriers(c, c["hidden"], c["mask"])
        for name in ("C", "P", "Q"):
            np.testing.assert_array_equal(result[name], [0, 0])
    with pytest.raises(ValueError):
        chart_carriers(charts[0], np.zeros((4, 4)), np.array([0, 0, 1, 0], bool))


def test_uniform_cannot_distinguish_token_permutation():
    rng = np.random.default_rng(2)
    r, x, g = (
        rng.normal(size=(5, 4)),
        rng.normal(size=(7, 4)),
        rng.normal(size=(2, 5, 4)),
    )
    np.testing.assert_allclose(
        pool_pairings(r, x, g, "uniform"),
        pool_pairings(r, x[::-1], g, "uniform"),
        atol=2e-6,
    )


def test_state_reconstruction_equals_direct_projection():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(12, 12)), rng.normal(size=(12, 12))
    model = fit_linear(x, y)
    f = FunctionalQuery(
        np.zeros((2, 4)), np.zeros(4), rng.normal(size=(2, 4)), rng.normal(size=4), 0.3
    )
    compiled = compile_affine(model, f)
    for early in rng.normal(size=(5, 12)):
        p, q = unpack_state(predict_linear(model, early), (2, 4))
        assert score_affine(compiled, early)["score"] == pytest.approx(
            f.parts(p, q)["score"], abs=1e-10
        )
