#!/usr/bin/env python3
"""Frozen reference, original addresses and material interval arithmetic."""

import json

import numpy as np
import pytest

from grain_research.common_factors import WordFactorReference
from grain_research.local_measure import (
    TOKEN_PATTERN,
    addressed_tokens,
    essay_blocks,
    explain_document,
    export_reference,
    material_interval,
    score_documents,
    word_rates,
)
from scripts.run_grain_local_construction import bh_adjust, correlation_row


@pytest.fixture
def bundle():
    return {
        "schema": "grain_local_lexical_reference_v1",
        "domain": "toy",
        "method": "pca",
        "token_pattern": TOKEN_PATTERN,
        "vocabulary": ["calm", "help"],
        "beta": [[2.0, -1.0], [1.0, 4.0]],
        "intercept": [-1.0, 0.5],
        "reference_sorted": [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
        "n_source": 3,
    }


def test_denominator_includes_unselected_words():
    rates, totals, _ = word_rates(["calm help unrelated the", ""], ["calm", "help"])
    np.testing.assert_allclose(rates[0], [0.25, 0.25])
    assert totals.tolist() == [4, 0]


def test_single_batch_and_duplicate_invariance(bundle):
    one = score_documents(bundle, ["calm help"])
    many = score_documents(bundle, ["unrelated " * 100, "calm help", "calm help"])
    np.testing.assert_array_equal(one["z"][0], many["z"][1])
    np.testing.assert_array_equal(one["percentile"][0], many["percentile"][2])


def test_empty_is_unknown_not_reference_mean(bundle):
    assert np.isnan(score_documents(bundle, ["123 !"])["z"]).all()
    assert explain_document(bundle, "")["score"] is None


def test_exact_word_explanation_with_unicode_prefix(bundle):
    text = "\u0130 said CALM, help."
    e = explain_document(bundle, text)
    assert e["max_error"] < 1e-12
    for hit in e["hits"]:
        assert text[hit["start"] : hit["end"]].lower() == hit["term"]
    assert list(addressed_tokens("\u0130 calm"))[-1] == ("calm", 2, 6)


def test_source_reference_export_matches_original_pca():
    rng = np.random.default_rng(17)
    vocab = [
        "calm",
        "help",
        "work",
        "friend",
        "music",
        "read",
        "build",
        "food",
        "sleep",
        "sad",
    ]
    docs = [
        " ".join(rng.choice(vocab, size=int(rng.integers(10, 35)))) for _ in range(40)
    ]
    model = WordFactorReference(method="pca", n_factors=3).fit(docs)
    b = export_reference(
        model, docs, domain="toy", source_ids=[str(i) for i in range(40)]
    )
    serialized = json.loads(json.dumps(b, allow_nan=False))
    expected = (model.transform(docs) - b["score_center"]) / b["score_scale"]
    np.testing.assert_allclose(
        expected, score_documents(serialized, docs)["z"], atol=1e-12
    )


def test_duplicate_source_ids_rejected():
    with pytest.raises(ValueError, match="unique author"):
        export_reference(None, ["a", "b"], domain="toy", source_ids=["x", "x"])


def test_material_draws_repeat_and_are_not_zero_for_single_block(bundle):
    assert material_interval(bundle, ["calm"]) is None
    a = material_interval(bundle, ["calm", "help", "other"])
    assert a == material_interval(bundle, ["calm", "help", "other"])
    assert a["scope"].startswith("conditional") and a["draws"] == 200


@pytest.mark.parametrize(
    "text", ["", "One.", "One. Two? Three! Four. Five", "a\nb\nc\nd"]
)
def test_essay_blocks_lossless(text):
    assert "".join(essay_blocks(text)) == text


def test_bh_and_degenerate_correlation():
    np.testing.assert_allclose(
        bh_adjust([0.01, 0.04, 0.03, 0.5]), [0.04, 0.0533333333, 0.0533333333, 0.5]
    )
    assert correlation_row(np.ones(10), np.arange(10))["r"] is None
    assert correlation_row(np.arange(10), np.arange(10))["r"] > 0.99


def test_invalid_empirical_reference_fails(bundle):
    bundle["reference_sorted"] = [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]
    with pytest.raises(ValueError, match="reference distribution"):
        score_documents(bundle, ["calm"])
