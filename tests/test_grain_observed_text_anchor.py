#!/usr/bin/env python3
"""Finite tests for observed anchors without persona-answer targets."""

import numpy as np
import pytest

from grain_research.construction import math_observed_text_anchor as M


def test_surface_counts_do_not_infer_state():
    definitions = [
        {"kind": "words", "forms": ["worried"]},
        {"kind": "negation"},
        {"kind": "punctuation", "symbol": "!"},
    ]
    counts, rates, words = M.lexical_observations(
        "I'm not worried! I don't worry.", definitions
    )
    assert counts.tolist() == [1, 2, 1]
    assert words == 6
    np.testing.assert_allclose(rates, 100 * counts / 6)


def test_conditional_cross_moments_keep_between():
    x = np.array([[0, 2], [2, 1], [6, 5], [8, 7]], float)
    y = np.array([[1, 0], [3, 1], [7, 2], [8, 4]], float)
    p = M.anchor_program(x, y, ["a", "a", "b", "b"])
    np.testing.assert_allclose(p["total"], p["within"] + p["between"], atol=1e-14)
    assert np.linalg.norm(p["between"]) > 0
    np.testing.assert_allclose(
        M.cold_score(p, x, "total"),
        M.cold_score(p, x, "within") + M.cold_score(p, x, "between"),
        atol=1e-14,
    )


def test_alternate_pairwise_covariance():
    rng = np.random.default_rng(164)
    x, y = rng.normal(size=(7, 5)), rng.normal(size=(7, 3))
    p = M.anchor_program(x, y, ["a"] * 3 + ["b"] * 4)
    pairs = sum((y[i] - y).T @ (x[i] - x) for i in range(len(x))) / (2 * len(x) ** 2)
    np.testing.assert_allclose(p["total"], pairs, atol=1e-14)


def test_application_observations_cannot_change_source_program():
    x, y = np.arange(15).reshape(5, 3), np.arange(10).reshape(5, 2).astype(float)
    mask = np.array([True, True, True, False, False])
    z, center, scale, _ = M.source_units(y, mask)
    before = M.anchor_program(x[mask], z[mask], ["a", "a", "b"])
    y[~mask] += 10000
    changed, center2, scale2, _ = M.source_units(y, mask)
    after = M.anchor_program(x[mask], changed[mask], ["a", "a", "b"])
    np.testing.assert_array_equal(center, center2)
    np.testing.assert_array_equal(scale, scale2)
    np.testing.assert_array_equal(before["total"], after["total"])


def test_absent_anchor_is_unobserved_not_zero_score():
    p = M.anchor_program(np.arange(12).reshape(4, 3), np.ones((4, 2)), ["a"] * 4)
    assert not p["supported"].any()
    assert np.isnan(M.cold_score(p, np.ones((1, 3)), "total")).all()


def test_pairing_rejects_missing_or_duplicate_observation():
    lo = {
        "axis": "e",
        "family": "f",
        "situation": "t",
        "style": "s",
        "seed": 1,
        "role": "source",
        "pole": -1,
    }
    hi = lo | {"pole": 1}
    pairs, _ = M.persona_pairs([lo, hi])
    np.testing.assert_array_equal(pairs, [[1, 0]])
    with pytest.raises(ValueError, match="Missing"):
        M.persona_pairs([lo])
    with pytest.raises(ValueError, match="Duplicate"):
        M.persona_pairs([lo, hi, hi])


def test_zero_and_reversed_effects_are_separate():
    result = M.signed_comparison([1, 0, -1, 1], [2, 1, 2, np.nan])
    assert result == {
        "same": 1,
        "opposite": 1,
        "observable_zero": 1,
        "reading_zero": 0,
        "unavailable": 1,
    }
