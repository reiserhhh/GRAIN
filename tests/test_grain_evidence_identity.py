#!/usr/bin/env python3
"""Synthetic contract tests; no participant text or model weights are required."""

import numpy as np
import pytest

from grain_research.evidence_identity import (
    MODES,
    QUERIES,
    address,
    evidence_record,
    opportunity_standardize,
    requests,
    simplex_decode,
    target_identifiability,
)
from scripts.run_grain_evidence_identity import public_bank


def test_bank_disjoint_and_complete():
    bank = public_bank()
    assert len(bank) == 52
    assert bank.record_id.is_unique
    assert bank.reply[bank.part.isin(["source", "held"])].is_unique
    assert bank[bank.part == "source"].groupby("expected").size().eq(2).all()
    assert bank[bank.part == "held"].groupby("expected").size().eq(2).all()


def test_requests_do_not_receive_truth():
    req = requests("a", "Yes.", "Did you ask for help?")
    assert len(req) == len(QUERIES)
    assert all("expected" not in r for r in req)
    bare = requests("a", "Yes.", "Did you ask for help?", include_context=False)
    assert all("Did you" not in r["sentence"] for r in bare)


def test_same_reply_distinct_context():
    a = requests("a", "Yes.", "Would you like help?")
    b = requests("b", "Yes.", "Did you ask for help?")
    assert a[0]["sentence"] != b[0]["sentence"]
    assert a[0]["item_text"] == b[0]["item_text"]


def test_unicode_address_and_multilabel():
    reply = "助けてください。"
    item = evidence_record("a", reply, "", {k: [0.8, 0.1, 0.1] for k in QUERIES})
    assert item["utterance"]["end"] == len(reply)
    assert sum(c["entailment"] for c in item["channels"].values()) > 1
    assert item["real_world_action"] == "unobserved"
    assert item["psychological_score"] is None
    assert item["antecedent"] is None


def test_antecedent_not_claimed_resolved():
    item = evidence_record("a", "Yes", "Help?", {k: [0.3, 0.6, 0.1] for k in QUERIES})
    assert item["antecedent"]["role"] == "partner"
    assert "not_resolved" in item["antecedent_status"]


@pytest.mark.parametrize("triple", [[0.8, 0.8, 0.8], [-1, 1, 1], [np.nan, 0, 1]])
def test_bad_probability(triple):
    with pytest.raises(ValueError):
        evidence_record("a", "Hi", "", {k: triple for k in QUERIES})


def test_decoder_and_rank():
    b = np.eye(6)
    p = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
    decoded, residual = simplex_decode(p, b)
    np.testing.assert_allclose(decoded, p, atol=1e-7)
    assert residual < 1e-7
    info = target_identifiability(b, [-1, 1, 0, 0, 0, 0])
    assert info["rank"] == len(MODES)
    assert info["target_identifiable_in_source_linear_model"]


def test_content_null_direction():
    b = np.ones((6, 1))
    q = np.array([-1, 1, 0, 0, 0, 0])
    info = target_identifiability(b, q)
    assert info["rank"] == 1
    assert not info["target_identifiable_in_source_linear_model"]
    p = np.full(6, 1 / 6)
    np.testing.assert_allclose((p + 0.1 * q) @ b, (p - 0.1 * q) @ b)
    assert q @ (p + 0.1 * q) != q @ (p - 0.1 * q)


def test_opportunity_mix_not_author_difference():
    f = [0.8, 0.2]
    assert opportunity_standardize(f, [0.9, 0.1]) == pytest.approx(0.74)
    assert opportunity_standardize(f, [0.1, 0.9]) == pytest.approx(0.26)
    assert opportunity_standardize(f, [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        opportunity_standardize(f, [1, 1])


def test_address_role_explicit():
    with pytest.raises(ValueError):
        address("hi", "1", "guessed")
