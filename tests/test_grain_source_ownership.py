#!/usr/bin/env python3
"""Ownership routes, source order and missing observations do not need an LLM."""

import pytest

from grain_research.source_ownership import (
    classify_source,
    quote_premise,
    select_observations,
)


def distribution(winner):
    """Produce a valid illustrative classifier result, not a calibrated posterior."""
    return {
        name: 0.8 if name == winner else 0.1
        for name in ("entailment", "contradiction", "neutral")
    }


@pytest.mark.parametrize(
    "own,other,expected",
    [
        ("entailment", "contradiction", "self"),
        ("contradiction", "entailment", "other"),
        ("entailment", "entailment", "unresolved"),
        ("neutral", "neutral", "unresolved"),
        ("contradiction", "contradiction", "unresolved"),
    ],
)
def test_two_claim_decision(own, other, expected):
    assert (
        classify_source({"self": distribution(own), "other": distribution(other)})[
            "source_owner"
        ]
        == expected
    )


def test_tied_entailment_abstains():
    assert (
        classify_source(
            {
                "self": {"entailment": 0.5, "neutral": 0.5, "contradiction": 0},
                "other": distribution("neutral"),
            }
        )["source_owner"]
        == "unresolved"
    )


@pytest.mark.parametrize("bad", [-0.1, float("nan"), 1.5])
def test_invalid_probabilities_rejected(bad):
    own = distribution("entailment")
    own["entailment"] = bad
    with pytest.raises(ValueError):
        classify_source({"self": own, "other": distribution("contradiction")})


def test_missing_claim_rejected():
    with pytest.raises(ValueError):
        classify_source({"self": distribution("entailment")})


def test_exact_passage_required():
    text = 'A friend said: "I keep checking the door."'
    premise = quote_premise(text, "I keep checking the door.")
    assert text in premise
    with pytest.raises(ValueError):
        quote_premise(text, "I never check the door.")


def test_selection_retains_context_and_order():
    texts = ["own1", "friend", "unclear", "own2"]
    result = select_observations(
        texts, [{"source_owner": s} for s in ("self", "other", "unresolved", "self")]
    )
    assert result["selected_texts"] == ["own1", "own2"]
    assert result["selected_indices"] == [0, 3]
    assert result["context_indices"] == [1, 2]
    assert result["all_texts_retained"] == texts
    assert result["excerpt_coverage"] == 0.5


def test_no_self_is_unavailable_not_zero():
    result = select_observations(
        ["friend", "unclear"],
        [{"source_owner": "other"}, {"source_owner": "unresolved"}],
    )
    assert result["status"] == "NO_OBSERVED_SELF_EVIDENCE"
    assert result["score_when_unobserved"] is None
    assert result["selected_texts"] == []
    assert result["all_texts_retained"] == ["friend", "unclear"]


def test_designed_labels_cannot_override_detector():
    result = select_observations(
        ["x"], [{"source_owner": "other", "expected_owner": "self", "oracle": True}]
    )
    assert result["status"] == "NO_OBSERVED_SELF_EVIDENCE"


@pytest.mark.parametrize(
    "texts,decisions",
    [
        ([], []),
        (["x"], []),
        ([""], [{"source_owner": "self"}]),
        (["x"], [{"source_owner": "invented"}]),
    ],
)
def test_invalid_selection_rejected(texts, decisions):
    with pytest.raises(ValueError):
        select_observations(texts, decisions)
