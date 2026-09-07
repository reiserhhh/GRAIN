#!/usr/bin/env python3
"""Matched-query and functional tensor tests without private data."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.cross_query_functional import (
    QUERIES,
    QUESTIONS,
    cosine_matrix,
    cross_readers,
    effect_quality,
    matched_prompt,
)
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from scripts.run_grain_multiquery_sensitivity import NARROW_QUESTION
from scripts.run_grain_numeric_query_probe import CONTROL_QUESTION


class TinyTokenizer:
    """Injective character codec for template and boundary equality tests."""

    def apply_chat_template(self, messages, **_kwargs):
        return list(
            (
                "SYS:"
                + messages[0]["content"]
                + "USR:"
                + messages[1]["content"]
                + "GEN:"
            ).encode()
        )


def test_native_replay_and_only_question_changes():
    """Even an excerpt containing the native question must remain untouched."""
    tokenizer = TinyTokenizer()
    text = ["An ordinary short comment.", QUESTIONS["neuroticism"]]
    original = tokenize_prompt(text, "neuroticism", tokenizer)
    assert matched_prompt(text, "neuroticism", tokenizer) == original
    for query in QUERIES:
        _ids, user, spans = matched_prompt(text, query, tokenizer)
        assert [user[a:b] for a, b in spans] == text
        assert user.rpartition("\nQuestion: ")[2] == QUESTIONS[query] + "\nAnswer:"
    assert QUESTIONS["confirmation_seeking"] == NARROW_QUESTION
    assert QUESTIONS["quantitative_content"] == CONTROL_QUESTION


def test_cross_matrix_has_correct_recipient_and_reader_axes():
    """Distinct orthogonal queries expose accidental transposes and averaging."""
    g = np.repeat(np.eye(4)[None], 5, axis=0)
    d = np.arange(24, dtype=float).reshape(2, 4, 3)
    d = np.concatenate((d, np.ones((2, 4, 1))), axis=2)
    means, scores = cross_readers(g, d)
    np.testing.assert_allclose(means, np.eye(4))
    np.testing.assert_allclose(scores, d)
    np.testing.assert_allclose(cosine_matrix(means), np.eye(4))


def test_reference_independence_and_cross_query_error_identity():
    """Held observations do not change readers; the declared error split is exact."""
    rng = np.random.default_rng(21)
    g, d = rng.normal(size=(7, 4, 9)), rng.normal(size=(3, 4, 9))
    mean, scores = cross_readers(g, d)
    held_g = rng.normal(size=(3, 4, 9))
    effects = np.sum(held_g * d, axis=-1)
    for q in range(4):
        for r in range(4):
            rhs = np.sum((held_g[:, q] - mean[q]) * d[:, q], axis=-1)
            rhs += d[:, q] @ (mean[q] - mean[r])
            np.testing.assert_allclose(effects[:, q] - scores[:, q, r], rhs)
    d[0] *= 100
    again, predictions = cross_readers(g, d)
    np.testing.assert_array_equal(mean, again)
    np.testing.assert_allclose(scores[1:], predictions[1:])


@pytest.mark.parametrize("d", [np.ones((3, 4)), np.full((3, 4, 9), np.nan)])
def test_bad_crossed_vectors_fail(d):
    """Reject incompatible or nonfinite geometry."""
    with pytest.raises(ValueError):
        cross_readers(np.ones((7, 4, 9)), d)


def test_zero_effect_is_reported_not_dropped():
    """A null mechanism still has measurable absolute error, not a fake r."""
    scores = effect_quality(np.zeros(4), np.ones(4))
    assert scores["r"] is None
    assert scores["mse_over_effect_energy"] is None
    assert scores["mse"] == 1
