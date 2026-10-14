"""Regression checks for the original F02 worlds, not merely output counts."""

from collections import Counter

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import (
    fit_observed_source,
    selection_views,
    sequential_predictions,
)


def test_c3_contains_both_ambiguous_types_with_matched_publication():
    records, truth = generate_world(
        "c3",
        20260915,
        events=1024,
        counts={"source": 40, "validation": 0, "application": 0},
    )
    by_id = {row["author_id"]: row for row in truth}
    assert {(row["pi"], row["q"]) for row in truth} == {(0.8, 0.5), (0.4, 1.0)}
    for kind, pi, q in [(-1, 0.8, 0.5), (1, 0.4, 1.0)]:
        rows = [row for row in records if by_id[row["author_id"]]["kind"] == kind]
        opportunities = sum(row["opportunity"] for row in rows)
        selected = sum(row["selected"] for row in rows)
        assert abs(opportunities / len(rows) - pi) < 0.02
        assert abs(selected / opportunities - q) < 0.02
        assert abs(selected / len(rows) - 0.4) < 0.02
        assert all(row["observation"] is None for row in rows if not row["selected"])


def test_c1_types_balanced_and_role_randomness_independent():
    counts = {"source": 12, "validation": 6, "application": 6}
    first, truth = generate_world("c1", 20260915, events=8, counts=counts)
    second, _ = generate_world(
        "c1", 20260915, events=8, counts={**counts, "source": 30}
    )
    assert Counter(row["kind"] for row in truth if row["role"] == "source") == {
        -1: 6,
        1: 6,
    }

    # Global event addresses may change; application observations must not.
    def observed(rows):
        return [
            (r["author_id"], r["t"], r["observation"])
            for r in rows
            if r["role"] == "application"
        ]

    assert observed(first) == observed(second)
    assert not any("kind" in row or "pi" in row or "q" in row for row in first)


def test_c1_conditional_responses_follow_declared_law():
    rows, truth = generate_world(
        "c1",
        20260916,
        events=2048,
        counts={"source": 12, "validation": 0, "application": 0},
    )
    kind = {row["author_id"]: row["kind"] for row in truth}
    for k in [-1, 1]:
        for z in [-1, 1]:
            values = [
                row["observation"]
                for row in rows
                if kind[row["author_id"]] == k and row["condition"]["z"] == z
            ]
            assert abs(np.mean(values) - (0.5 + 0.25 * k * z)) < 0.025


def test_source_rejects_evaluation_and_truth():
    rows, _ = generate_world(
        "c1",
        20260915,
        events=8,
        counts={"source": 12, "validation": 6, "application": 6},
    )
    with pytest.raises(ValueError, match="source-only"):
        fit_observed_source(rows, "c1")
    source = [row for row in rows if row["role"] == "source"]
    with pytest.raises(ValueError, match="truth-bearing"):
        fit_observed_source([{**source[0], "kind": 1}], "c1")


@pytest.mark.parametrize("world", ["c1", "c2_xor", "c2_iid"])
def test_predict_before_future_is_seen(world):
    rows, _ = generate_world(
        world,
        20260915,
        events=8,
        counts={"source": 12, "validation": 0, "application": 6},
    )
    fit = fit_observed_source([r for r in rows if r["role"] == "source"], world)
    application = [r for r in rows if r["role"] == "application"]
    original = sequential_predictions(application, fit, 4)
    changed = sequential_predictions(
        [
            {**r, "observation": 1 - r["observation"]} if r["t"] >= 4 else r
            for r in application
        ],
        fit,
        4,
    )
    assert [r["p"] for r in original if r["t"] == 4] == [
        r["p"] for r in changed if r["t"] == 4
    ]
    # Future labels may affect later sequential updates, never the first prediction.


def test_opportunity_masked_view_does_not_depend_on_opportunity():
    rows, _ = generate_world(
        "c3",
        20260915,
        events=8,
        counts={"source": 0, "validation": 0, "application": 6},
    )
    first = selection_views(rows, (4, 8))
    second = selection_views([{**r, "opportunity": 1} for r in rows], (4, 8))
    assert [r for r in first if r["view"] == "opportunity_masked"] == [
        r for r in second if r["view"] == "opportunity_masked"
    ]
