#!/usr/bin/env python3
"""Pure-bias null, sigmoid artifact, semantic recoding and interaction controls."""

import copy

import numpy as np
import pytest

from grain_research.vector_mechanisms.direct_encoding_effects import (
    aligned_logits,
    analyze,
)
from grain_research.vector_mechanisms.direct_semantics import panel
from grain_research.vector_mechanisms.direct_traits import BIG5


def additive_rows():
    """Build all cases from an exactly content-independent coding-bias model."""
    rows = []
    for c in panel():
        size = 10 if c["task"] in BIG5 else 2
        v = np.linspace(0, 1, size)
        if c["task"] in BIG5 and c["variant"] == "counter":
            v = 1 - v
        x = (
            0
            if c["pair"] < 0
            else (-0.7 if c["polarity"] == 0 else 0.8) + 0.3 * c["pair"]
        )
        direction = np.linspace(-1, 1, size)
        bias = 1.7 * direction if c["variant"] == "counter" else 0.0
        semantic = x * direction + bias
        raw = np.empty(size)
        raw[np.argsort(v)] = semantic
        rows.append(
            c | {"candidate_logits": raw.tolist(), "alphabet": {"values": v.tolist()}}
        )
    return rows


def test_constant_logit_bias_is_not_confused_with_sigmoid_curvature():
    rows = additive_rows()
    tasks, cases, pairs = analyze(rows)
    assert len(tasks) == 9 and len(cases) == 36 and len(pairs) == 18
    assert max(t["anchored_residual_rms"] for t in tasks) < 1e-14
    assert max(p["mixed_rms"] for p in pairs) < 1e-14
    ei = [r for r in rows if r["task"] == "EI_cont" and r["pair"] == 0]
    probability = {}
    for r in ei:
        z = aligned_logits(r)
        probability[r["variant"], r["polarity"]] = 1 / (1 + np.exp(-(z[1] - z[0])))
    diff = (
        probability["counter", 1]
        - probability["counter", 0]
        - probability["primary", 1]
        + probability["primary", 0]
    )
    assert abs(diff) > 0.1


def test_content_dependent_effect_has_reference_independent_mixed_contrast():
    rows = additive_rows()
    for r in rows:
        if (
            r["task"] == "EI_cont"
            and r["variant"] == "counter"
            and r["pair"] == 0
            and r["polarity"] == 1
        ):
            r["candidate_logits"][1] += 2.0
    _, _, pairs = analyze(rows)
    p = next(p for p in pairs if p["task"] == "EI_cont" and p["pair"] == 0)
    assert p["mixed_log_odds"] == pytest.approx(2.0)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["variant"] == "counter":
            r["candidate_logits"][0] -= 11
        if r["pair"] == -1:
            r["candidate_logits"] = (np.array(r["candidate_logits"]) * 3).tolist()
    _, _, other = analyze(changed)
    q = next(q for q in other if q["task"] == "EI_cont" and q["pair"] == 0)
    assert q["mixed_log_odds"] == pytest.approx(p["mixed_log_odds"])


def test_gauge_and_joint_answer_permutation_invariance():
    rows = additive_rows()
    expected = analyze(rows)
    rng = np.random.default_rng(52)
    shifted = copy.deepcopy(rows)
    for r in shifted:
        z = np.array(r["candidate_logits"]) + rng.uniform(-20, 20)
        v = np.array(r["alphabet"]["values"])
        perm = rng.permutation(len(z))
        r["candidate_logits"] = z[perm].tolist()
        r["alphabet"]["values"] = v[perm].tolist()
    actual = analyze(shifted)
    for a, e in zip(actual[2], expected[2], strict=True):
        np.testing.assert_allclose(a["mixed_vector"], e["mixed_vector"], atol=1e-13)
    for r in actual[1]:
        assert abs(sum(r["semantic_shift"])) < 1e-13


def test_changed_panel_or_invalid_semantic_coding_refused():
    rows = additive_rows()
    with pytest.raises(ValueError, match="grid"):
        analyze(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[0]["text"] = "Changed after observation"
    with pytest.raises(ValueError, match="grid"):
        analyze(bad)
    bad = copy.deepcopy(rows)
    bad[0]["alphabet"]["values"][1] = 0
    with pytest.raises(ValueError, match="coding"):
        analyze(bad)
