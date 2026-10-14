"""Paired-design, source-isolation and observation-view regression checks."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f03 import generate_records
from grain_research.gptpro_foundation_v3.f03_recovery import (
    fit_source_selection,
    observe_view,
    personal_propensity,
    score_views,
)


def test_designs_share_authors_covariates_and_observed_potential_responses():
    counts = {"source": 4, "validation": 2, "application": 2}
    generated = {
        design: generate_records("W3-D", design, 20260915, events=32, counts=counts)
        for design in ("free", "common", "fixed")
    }
    fixed, fixed_truth = generated["fixed"]
    for rows, truth in generated.values():
        assert [{k: v for k, v in t.items() if k != "design"} for t in truth] == [
            {k: v for k, v in t.items() if k != "design"} for t in fixed_truth
        ]
        for row, reference in zip(rows, fixed):
            assert row["author_id"] == reference["author_id"]
            assert row["condition"] == reference["condition"]
            if row["selected"]:
                assert row["observation"] == reference["observation"]
    assert [r["selected"] for r in generated["free"][0]] != [
        r["selected"] for r in generated["common"][0]
    ]


def test_nested_prefix_and_role_stream_independence():
    counts = {"source": 3, "validation": 2, "application": 2}
    short, _ = generate_records("W3-A", "free", 20260915, events=32, counts=counts)
    long, _ = generate_records(
        "W3-A", "free", 20260915, events=128, counts={**counts, "source": 8}
    )
    fields = ("author_id", "t", "condition", "selected", "opportunity", "observation")

    def comparable(rows):
        return [
            {k: r[k] for k in fields}
            for r in rows
            if r["role"] == "application" and r["t"] < 32
        ]

    assert comparable(short) == comparable(long)


def test_source_only_selection_recovers_known_mar_slopes():
    rows, _ = generate_records(
        "W3-B",
        "free",
        20260915,
        events=256,
        counts={"source": 48, "validation": 0, "application": 0},
    )
    model = fit_source_selection(rows)
    assert model["converged"]
    np.testing.assert_allclose(model["shared_slopes"], [0.8, 1.6], atol=0.2)
    assert model["intercept_variance"] >= 0.1
    with pytest.raises(ValueError, match="source-only"):
        fit_source_selection([{**rows[0], "role": "application"}])


def test_propensity_never_uses_selected_response_values():
    rows, _ = generate_records(
        "W3-D",
        "common",
        20260915,
        events=32,
        counts={"source": 6, "validation": 0, "application": 1},
    )
    source = fit_source_selection([r for r in rows if r["role"] == "source"])
    prefix = [r for r in rows if r["role"] == "application"]
    first = personal_propensity(prefix, source)
    second = personal_propensity([{**r, "observation": 999} for r in prefix], source)
    assert first["intercept"] == second["intercept"]
    np.testing.assert_array_equal(first["q"], second["q"])


def test_missing_denominator_is_removed_not_guessed():
    rows, _ = generate_records(
        "W3-C",
        "free",
        20260915,
        events=32,
        counts={"source": 0, "validation": 0, "application": 1},
    )
    published = observe_view(rows, "published_only")
    assert all(
        "opportunity" not in r and "t" not in r and "event_id" not in r
        for r in published
    )
    masked = observe_view(rows, "selected_indicator")
    assert all("opportunity" not in r for r in masked)
    scores = score_views(rows, None, "free")
    assert all(
        r["f"] is None and r["bound_low"] is None
        for r in scores
        if r["view"] == "published_only"
    )
    assert all(
        r["pi"] is None and r["q"] is None
        for r in scores
        if r["view"] != "all_opportunities"
    )


def test_personal_map_matches_independent_dense_grid():
    from scipy.special import expit

    rows, _ = generate_records(
        "W3-A",
        "common",
        20260915,
        events=32,
        counts={"source": 0, "validation": 0, "application": 1},
    )
    source = {
        "shared_slopes": [0, 1.6],
        "intercept_mean": -0.8,
        "intercept_variance": 0.3,
    }
    result = personal_propensity(rows, source)
    grid = np.linspace(-6, 6, 24001)
    offsets = np.array([1.6 * r["condition"]["x"] for r in rows])
    labels = np.array([r["selected"] for r in rows])
    probs = expit(grid[:, None] + offsets)
    losses = (
        -(labels * np.log(probs) + (1 - labels) * np.log1p(-probs)).sum(axis=1)
        + (grid + 0.8) ** 2 / 0.6
    )
    assert abs(result["intercept"] - grid[np.argmin(losses)]) < 0.001
