"""Synthetic algebra and immutable evidence checks for the M13 explanation chain."""

import copy

import numpy as np
import pytest

from grain_research.explanation_chain import (
    condition_profile,
    decompose_query,
    interpretation_scope,
    make_packet,
    public_bank,
)
from grain_research.query_moment_operator import apply_operator
from grain_research.vector_mechanisms.interpretation_evidence import validate_packet
from scripts.run_grain_within_comment_channels import lexical_slices


def test_bank_complete_and_lossless():
    """Keep all cells, both wordings and every raw character."""
    rows = public_bank()
    assert len(rows) == 24
    assert (
        len({(r["identity"], r["pole"], r["minutes"], r["wording"]) for r in rows})
        == 24
    )
    for row in rows:
        parts = lexical_slices(row["text"])
        assert len(parts) == 4 and "".join(parts) == row["text"]
        assert row["text"].count(row["target_span"]) == 1


def test_potential_accounting():
    """Independent expansion equals the third reader including its own anchor."""
    rng = np.random.default_rng(91431)
    g = np.repeat(rng.normal(size=(4, 4)), 3, axis=0)
    h = rng.normal(size=(12, 7))
    a = rng.normal(size=(2, 2))
    model = {
        "g_mean": np.zeros(4),
        "g_scale": np.ones(4),
        "beta1": rng.normal(size=(5, 7)),
        "beta2": rng.normal(size=(15, 7)),
        "query_basis": rng.normal(size=(7, 2)),
        "within_matrix": a + a.T,
        "offsets": np.array([3.0, 9.0, -7.0]),
    }
    expected = apply_operator(g, h, model)[0][2]
    anchor, between, within = decompose_query(g, h, model)
    assert anchor + between.sum() + within.sum() == pytest.approx(expected)


def test_condition_profile():
    """Finite half-change reconstructs both endpoints exactly."""
    row = condition_profile(-3, 7)
    assert row == {"level": 2, "half_change": 5}
    with pytest.raises(ValueError):
        condition_profile(np.nan, 1)


@pytest.mark.parametrize("index", [0, 8, 16])
def test_immutable_bindings(index):
    """Scores and verbatim addresses cannot be changed by a renderer."""
    row = public_bank()[index]
    values = {"grain_f03": 0.2, "query": -0.4}
    packet = make_packet(row, values, {k: "a" * 64 for k in values})
    trusted = {m["id"]: copy.deepcopy(m) for m in packet["measurements"]}
    assert validate_packet(packet, {row["case"]: row["text"]}, trusted)[
        "valid_bindings"
    ]
    bad = copy.deepcopy(packet)
    bad["measurements"][0]["value"] += 1
    with pytest.raises(ValueError):
        validate_packet(bad, {row["case"]: row["text"]}, trusted)
    bad = copy.deepcopy(packet)
    bad["evidence"][1]["quote"] += " invented"
    with pytest.raises(ValueError):
        validate_packet(bad, {row["case"]: row["text"]}, trusted)


def test_identity_is_not_score():
    """Identical scores cannot turn an intention into an observed action."""
    values = {"grain_f03": 10, "query": 10}
    packets = [
        make_packet(public_bank()[i], values, {k: "b" * 64 for k in values})
        for i in [0, 8, 16]
    ]
    assert len({p["candidates"][0]["hypothesis"] for p in packets}) == 3
    assert interpretation_scope("reported_action")[0] == "reported_event"
    assert all(p["candidates"][0]["human_status"] == "not_reviewed" for p in packets)
