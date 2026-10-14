#!/usr/bin/env python3
"""Finite mechanism tests, not prediction-accuracy gates."""

from itertools import combinations

import numpy as np
import pytest

from grain_research.construction import math_prospective_evidence as H
from grain_research.construction import math_selector_matrix as M


def fixture():
    """Construct complete SOURCE support without human/private data."""
    rng = np.random.default_rng(248)
    codes = np.repeat(np.arange(32), 2)
    z = rng.normal(size=(64, 2, 5)) * 0.2
    z[:, :, 2] += ((codes >> 2) & 1)[:, None]
    bank = {"source_means": z, "codes": codes, "variance": 0.7}
    x = rng.normal(size=(12, 5))
    subsets = np.array(list(combinations(range(12), 4)))
    accounts = H.accounts(bank, (x.sum(0) - x[subsets].sum(1)) / 8)
    return bank, x, subsets, accounts


def test_exact_old_A_rule_and_all_matches():
    bank, x, s, a = fixture()
    c, length = np.arange(12) % 2, np.full(12, 10)
    old = H.select(bank, x, c, length)
    rows = M.select_all(a, s, c, length)
    assert (rows[2]["support_index"], rows[2]["counter_index"]) == (
        old["support"],
        old["counter"],
    )
    for q, row in enumerate(rows):
        assert not set(row["support"]) & set(row["counter"])
        assert row["removed_condition_counts"][0] == row["removed_condition_counts"][1]
        assert row["source_action"][2][q] >= 0


def test_ties_are_lexicographic_and_zero_not_rescued():
    _, _, s, _ = fixture()
    rows = M.select_all(np.zeros((495, 3, 5)), s, np.zeros(12), np.ones(12, int))
    assert rows[0]["support"] == [0, 1, 2, 3]
    assert rows[0]["counter"] == [4, 5, 6, 7]
    assert rows[0]["source_action"] == np.zeros((3, 5)).tolist()


def test_no_matching_fallback_or_missing_target():
    _, _, s, a = fixture()
    with pytest.raises(ValueError):
        M.select_all(a, s, np.zeros(12), 2 ** np.arange(12), absolute=0, relative=0)
    with pytest.raises(ValueError):
        M.select_all(a[..., :4], s, np.zeros(12), np.ones(12, int))


def test_fixed_evaluation_can_be_large_but_primary_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(24, 5, 1)) * M.EVALUATIVE
    result = M.decompose(x)
    assert np.max(abs(result["primary"])) < 1e-15
    assert np.max(abs(result["interaction"])) < 1e-15
    assert np.allclose(result["evaluative"] + result["remainder"], x)


def test_matched_target_signal_not_removed_with_evaluation():
    x = np.broadcast_to(np.eye(5), (24, 5, 5)).copy()
    result = M.decompose(x)
    assert np.allclose(result["primary"], 0.8)
    assert np.allclose(result["remainder"] @ M.EVALUATIVE, 0)
    contact, _ = M.crossbank_witness(result["interaction"])
    assert contact > 0


def test_opposite_bank_contact_is_signed_not_squared():
    x = np.ones((24, 5, 5))
    x[12:] *= -1
    contact, _ = M.crossbank_witness(x)
    assert contact == -1


def test_bootstrap_resamples_authors_and_is_deterministic():
    x = np.arange(24)[:, None, None] * np.ones((1, 5, 5))
    ci = M.bootstrap(x, 99, 2)
    assert ci.shape == (5, 5, 2)
    assert np.array_equal(ci, M.bootstrap(x, 99, 2))
    assert np.array_equal(ci[0, 0], ci[4, 4])


def test_digit_gauge_invariance_and_complete_shape():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(24, 5, 2, 5, 10))
    y, d = M.read_actions(logits)
    yz, dz = M.read_actions(logits + rng.normal(size=(24, 5, 2, 5, 1)))
    assert np.allclose(y, yz)
    assert np.allclose(d, dz)
    with pytest.raises(ValueError):
        M.read_actions(logits[:, :4])


def test_parser_requires_complete_nonambiguous_old_excerpts():
    text = "The excerpts below were written by one anonymous writer.\n\n"
    bodies = [f"body {i}" for i in range(12)]
    text += "\n\n".join(f"Excerpt {i + 1}:\n{b}" for i, b in enumerate(bodies))
    text += "\nEvidence ends here.\n\nQuestion: fixed"
    assert M.extract_bodies(text) == bodies
    with pytest.raises(ValueError):
        M.extract_bodies(text.replace("Excerpt 12:", "Excerpt 13:"))
    with pytest.raises(ValueError):
        M.extract_bodies(text + "\nEvidence ends here.\n\nQuestion: duplicate")
