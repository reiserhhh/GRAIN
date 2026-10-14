#!/usr/bin/env python3
"""Finite endpoint and source-action checks without neural calls."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction.math_endpoint_action import (
    endpoint_scores,
    natural_partner,
    policy_contrast,
    questions,
)
from grain_research.construction.math_qualified_target_field import (
    characters,
    reconstruct,
)


def test_complete_queries_preserve_person_target_and_instruction():
    """No endpoint family silently erases an actor or source target."""
    definitions, texts = questions()
    assert len(definitions) == len(texts) == 16
    assert len({tuple(d.values()) for d in definitions}) == 16
    for d, text in zip(definitions, texts, strict=True):
        assert ("Person A" if d["person"] == 1 else "Person B") in text
        if d["family"] == "prospective":
            assert "role-play" in text
            assert "extraversion" not in text and "neuroticism" not in text
        if d["family"].startswith("forced_"):
            assert "regardless of the record or your choice" in text


def test_full_mass_and_binary_policy_are_not_interchangeable():
    """A constant binary preference can coexist with different response mass."""
    logits = np.array([[2.0, 1.0], [2.0, 1.0]])
    norm = logsumexp(np.array([[2.0, 1.0, -10.0], [2.0, 1.0, 9.0]]), axis=1)
    result = endpoint_scores(logits, norm)
    assert result["restricted_yes"][0] == result["restricted_yes"][1]
    assert result["answer_mass"][0] > 0.99
    assert result["answer_mass"][1] < 0.01
    np.testing.assert_allclose(
        result["full_yes"], result["answer_mass"] * result["restricted_yes"]
    )


def test_probability_gauge_and_answer_reversal():
    """Common logit shifts leave probabilities fixed; swapping labels reverses D."""
    logits = np.array([[0.3, -0.8], [-3.0, 2.0]])
    norm = logsumexp(np.column_stack((logits, [1.0, 1.0])), axis=1)
    a, b = endpoint_scores(logits, norm), endpoint_scores(logits + 9, norm + 9)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-14)
    swapped = endpoint_scores(logits[:, ::-1], norm)
    np.testing.assert_allclose(swapped["log_odds"], -a["log_odds"])
    np.testing.assert_allclose(swapped["restricted_yes"], 1 - a["restricted_yes"])


@pytest.mark.parametrize("bad", [np.array([[1, 2, 3]]), np.array([[np.nan, 0]])])
def test_invalid_endpoint_fails_closed(bad):
    """Missing or nonfinite logits cannot become zero scores."""
    with pytest.raises(ValueError):
        endpoint_scores(bad, np.array([3]))


def test_invalid_probability_mass_fails_closed():
    """An invalid vocabulary normalizer is not treated as a perfect endpoint."""
    with pytest.raises(ValueError):
        endpoint_scores([[2, 2]], [1])


def test_actor_exchange_holds_opportunity_and_order():
    """The natural reference changes only the registered actor assignment."""
    row = {"condition": 1, "order": 0, "e": -1, "n": 1}
    indexed = {(1, 0, 1, 1): 4, (1, 0, -1, -1): 5}
    assert natural_partner(row, "e", indexed) == 4
    assert natural_partner(row, "n", indexed) == 5
    with pytest.raises(ValueError):
        natural_partner(row, "unknown", indexed)


def test_instructions_can_hide_an_existing_source_action():
    """A policy-controlled endpoint can suppress action without changing its source."""
    result = policy_contrast([1, -1], [0, 0], [0, 0])
    assert result["free_rms"] == 1
    assert result["forced_yes_rms"] == result["forced_no_rms"] == 0
    assert result["free_minus_forced_mean_rms"] == 1


def test_keep_full_joint_character_not_just_named_axis():
    """All relational terms survive the finite transform exactly."""
    coefficients = np.arange(8, dtype=float)
    np.testing.assert_allclose(characters(reconstruct(coefficients)), coefficients)
