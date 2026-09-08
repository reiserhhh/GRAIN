#!/usr/bin/env python3
"""Exact expanded score/response accounting on synthetic nine-query states."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from scripts.analyze_grain_all_factor_functions import (
    QUERIES,
    covariance_atlas,
    summarize_states,
)


def test_nine_query_response_and_covariance_accounting():
    rng = np.random.default_rng(7)
    functions = {
        q: FunctionalQuery(
            np.zeros((3, 2)),
            np.zeros(2),
            rng.normal(size=(3, 2)),
            rng.normal(size=2),
            0.4,
        )
        for q in QUERIES
    }
    rows = []
    for u in range(8):
        for view in (0, 1):
            for q in QUERIES:
                origin_p, origin_q = rng.normal(size=(3, 2)), rng.normal(size=2)
                for c in ("full", "e1", "e2"):
                    P = origin_p + (0 if c == "full" else rng.normal(size=(3, 2)))
                    Q = origin_q + (0 if c == "full" else rng.normal(size=2))
                    value = functions[q].parts(P, Q)["score"]
                    d = {
                        "author_id": f"test{u}",
                        "view": view,
                        "query": q,
                        "condition": c,
                        "role": "reference" if u < 4 else "evaluation",
                        "native_score": value,
                    }
                    rows.append((d, P, Q))
    frame, events = summarize_states(rows, functions)
    np.testing.assert_allclose(frame.score, frame.native_score, atol=1e-12)
    np.testing.assert_allclose(
        events.score, events[["mean", "configuration", "final"]].sum(axis=1), atol=1e-12
    )
    summary, detail = covariance_atlas(events)
    assert len(summary) == 36 * 5 and len(detail) == 36 * 5 * 16
    assert summary.accounting_error.max() < 1e-12
    np.testing.assert_allclose(summary.native_r, summary.functional_r, atol=1e-12)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_states(rows + rows[:1], functions)
