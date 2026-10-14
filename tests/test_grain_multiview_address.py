#!/usr/bin/env python3
"""Synthetic regression checks for original-address replay and author ranking."""

import numpy as np
import pandas as pd
import pytest

from grain_research.multiview_address import (
    align_units,
    author_rank_auc,
    sentence_coverage,
    token_index,
)
from suica_core.v7_observations import (
    ObservationSpec,
    build_observations,
    canonicalize_comments,
)


def fixture(kind, **kwargs):
    """Create repeated phrases and non-ASCII raw offsets deliberately."""
    body = "  same phrase.\n same phrase.\n 日本語 word! last tail. "
    raw = {0: {"author": "u", "body": body}}
    frame = canonicalize_comments(pd.DataFrame(raw.values()), min_tokens=1)
    frame["split"] = "test"
    spec = ObservationSpec(
        name=kind,
        kind=kind,
        min_tokens=1,
        max_units_per_user=128,
        max_source_comments_per_user=32,
        max_source_tokens_per_user=4096,
        **kwargs,
    )
    return raw, spec, build_observations(frame, spec)


@pytest.mark.parametrize(
    "kind,kw",
    [
        ("native", {}),
        ("whole", {}),
        ("semantic", {"max_tokens": 4}),
        ("fixed", {"window_tokens": 3, "stride_tokens": 3}),
        (
            "fixed",
            {"window_tokens": 3, "stride_tokens": 3, "source_boundary_mode": "within"},
        ),
    ],
)
def test_repeated_phrase_roundtrip(kind, kw):
    raw, spec, units = fixture(kind, **kw)
    ix = token_index(raw)
    segments = align_units(units, ix, spec)
    assert segments.iloc[0].char_start == 2
    assert segments.token_end.max() == len(ix[0]["tokens"])
    if len(segments) > 1:
        assert (np.diff(segments.char_start) > 0).all()


def test_wrong_text_fails_closed():
    raw, spec, units = fixture("native")
    units.loc[0, "text"] = units.loc[0, "text"].replace("same", "fake", 1)
    with pytest.raises(ValueError, match="sequence"):
        align_units(units, token_index(raw), spec)


def test_overlap_rejected():
    raw, spec, units = fixture("fixed", window_tokens=4, stride_tokens=2)
    with pytest.raises(ValueError, match="nonoverlapping"):
        align_units(units, token_index(raw), spec)


def test_partial_sentence_is_not_missing_evidence():
    raw, spec, units = fixture("fixed", window_tokens=2, stride_tokens=2)
    ix = token_index(raw)
    seg = align_units(units, ix, spec)
    sentence = {"source_row": 0, "start": 2, "end": 14, "text": raw[0]["body"][2:14]}
    out = sentence_coverage([sentence], seg, ix).iloc[0]
    assert out.retention == 1
    assert out.one_unit_completeness < 1
    assert out.touching_units == 2


def test_rank_perfect_ties_and_identity_exclusion():
    perfect = author_rank_auc(np.eye(6), np.eye(6), draws=100)
    assert perfect["auc"] == perfect["ci_low"] == 1
    tied = author_rank_auc(np.ones((6, 3)), np.ones((6, 3)), draws=100)
    assert tied["auc"] == tied["ci_high"] == 0.5


def test_rank_invariance_and_zero_rows():
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(10, 4)), rng.normal(size=(10, 4))
    a = author_rank_auc(x, y, draws=50)
    b = author_rank_auc(y * 3, x * 5, draws=50)
    assert a["auc"] == b["auc"]
    c = author_rank_auc(
        np.vstack([x, np.zeros(4)]), np.vstack([y, np.zeros(4)]), draws=50
    )
    assert c["auc"] == a["auc"] and c["zero_norm_excluded"] == 1


def test_wrong_author_and_weights_rejected():
    raw, spec, units = fixture("native")
    bad = units.copy()
    bad.loc[0, "user_id"] = "other"
    with pytest.raises(ValueError, match="Cross-author"):
        align_units(bad, token_index(raw), spec)
    bad = units.copy()
    bad.loc[0, "source_token_weights"] = "1"
    with pytest.raises(ValueError, match="weights"):
        align_units(bad, token_index(raw), spec)


def test_source_cap_returns_partial_coverage():
    from dataclasses import replace

    raw, spec, _ = fixture("native")
    spec = replace(spec, max_source_tokens_per_user=4)
    frame = canonicalize_comments(pd.DataFrame(raw.values()), min_tokens=1)
    frame["split"] = "test"
    ix = token_index(raw)
    seg = align_units(build_observations(frame, spec), ix, spec)
    s = {
        "source_row": 0,
        "start": 2,
        "end": len(raw[0]["body"]),
        "text": raw[0]["body"][2:],
    }
    actual = sentence_coverage([s], seg, ix).iloc[0]
    assert actual.retained_tokens == 4
    assert 0 < actual.retention < 1
    assert not actual.full_retention


def test_cross_comment_records_stay_separate():
    raw = {
        0: {"author": "u", "body": "one two"},
        1: {"author": "u", "body": "one two three"},
    }
    frame = canonicalize_comments(pd.DataFrame(raw.values()), min_tokens=1)
    frame["split"] = "test"
    spec = ObservationSpec(
        "cross", "fixed", window_tokens=4, stride_tokens=4, min_tokens=1
    )
    seg = align_units(build_observations(frame, spec), token_index(raw), spec)
    first = seg[seg.unit_index == 0]
    assert first.source_row.tolist() == [0, 1]
    assert first.token_start.tolist() == [0, 0]
    assert seg.iloc[-1].token_start == 2


def test_bootstrap_matches_explicit_author_draws():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    actual = author_rank_auc(x, y, draws=31, seed=16)
    sim = (x / np.linalg.norm(x, axis=1)[:, None]) @ (
        y / np.linalg.norm(y, axis=1)[:, None]
    ).T
    draws = np.random.default_rng(16).multinomial(7, np.full(7, 1 / 7), size=31)
    estimates = []
    for weights in draws:
        sample = np.repeat(np.arange(7), weights)
        if len(set(sample)) < 2:
            continue
        anchors = []
        for i in sample:
            others = sample[sample != i]
            both = []
            for matrix in (sim, sim.T):
                both.append(np.mean((matrix[i, i] > matrix[i, others]).astype(float)))
            anchors.append(np.mean(both))
        estimates.append(np.mean(anchors))
    np.testing.assert_allclose(
        [actual["ci_low"], actual["ci_high"]],
        np.quantile(estimates, [0.025, 0.975]),
        atol=1e-12,
    )
