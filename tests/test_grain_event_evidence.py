#!/usr/bin/env python3
"""Reader polarity, event multiplicity and missingness checks on public fixtures."""

import numpy as np
import pandas as pd
import pytest

from grain_research.event_evidence import (
    aggregate_events,
    conditional_profile,
    paired_evidence,
    split_controlled_reported_speech,
)


def fixture():
    """Return two known episodes from one anonymous synthetic author."""
    return pd.DataFrame(
        [
            {
                "author_id": "s0",
                "query": "q",
                "event_id": "e0",
                "condition": -1,
                "sentence_hash": "a",
                "score": -0.8,
                "accepted": True,
            },
            {
                "author_id": "s0",
                "query": "q",
                "event_id": "e1",
                "condition": 1,
                "sentence_hash": "b",
                "score": 0.8,
                "accepted": True,
            },
        ]
    )


def test_pair_contrast_and_abstention():
    """Weak, conflicting and contrary evidence are different outcomes."""
    result = paired_evidence([0.9, 0.1, 0.1, 0.8], [0.1, 0.9, 0.1, 0.7])
    np.testing.assert_allclose(result["direction"][:2], [1, -1])
    assert np.isnan(result["direction"][2:]).all()
    assert result["conflict"].tolist() == [False, False, False, True]
    with pytest.raises(ValueError):
        paired_evidence([1.1], [0])


def test_repeated_event_is_not_independent_evidence():
    """Repeated copies cannot increase the event count or alter the score."""
    frame = fixture()
    pd.testing.assert_frame_equal(
        aggregate_events(frame), aggregate_events(pd.concat([frame] * 4))
    )
    conflict = frame.copy()
    conflict.loc[0, "score"] = 0.8
    with pytest.raises(ValueError, match="Conflicting"):
        aggregate_events(pd.concat([frame, conflict]))


def test_missing_is_not_negative_or_zero():
    """No accepted evidence in one condition prevents a conditional contrast."""
    frame = fixture()
    frame.loc[0, "accepted"] = False
    aggregate = aggregate_events(frame)
    assert aggregate.iloc[0].coverage == 0
    assert np.isnan(aggregate.iloc[0].mean_score)
    assert np.isnan(conditional_profile(aggregate).iloc[0].response)


def test_opposite_responses_survive_zero_mean():
    """An author's balanced mean can vanish while the response is nonzero."""
    profile = conditional_profile(aggregate_events(fixture()))
    assert profile.iloc[0].level == 0
    assert profile.iloc[0].response == 0.8


def test_reported_speech_keeps_ownership_and_internal_apostrophe():
    """Keep the friend's evidence with exact offsets, not in the writer score."""
    text = "My friend told me: 'I enjoyed everyone's company.' My own experience was different. I disliked it."
    split = split_controlled_reported_speech(text)
    assert split["writer_text"] == " My own experience was different. I disliked it."
    span = split["other_spans"][0]
    assert (
        text[span["start"] : span["end"]]
        == span["text"]
        == "I enjoyed everyone's company."
    )
    unknown = "My colleague mentioned having enjoyed everyone's company."
    assert split_controlled_reported_speech(unknown)["writer_text"] == unknown


def test_end_to_end_aggregation_without_model_weights():
    """Exercise the runner's event joins and train/held profiles with oracle readers."""
    import json
    from pathlib import Path

    from scripts.run_grain_event_evidence_study import (
        assess_profiles,
        sentence_bank,
        simulate,
    )

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/grain_event_evidence.json"
        ).read_text()
    )
    config.update(authors=48, source_authors=24, events_per_author=8)
    bank = sentence_bank(config)
    for scope in ("episode", "habitual", "owner_episode"):
        bank[scope + "_contrast"] = bank.expected_direction.astype(float)
        bank[scope + "_accepted"] = bank.expected_direction.ne(0)
    bank["cosine_margin"] = bank.expected_direction * 0.1
    bank["cosine_key"] = np.where(bank.expected_direction >= 0, 1, -1)
    events, truth = simulate(config, bank, 731, "response")
    assert events[events.author_id == "synthetic_000"].variant.max() < 2
    assert events[events.author_id == "synthetic_047"].variant.min() >= 2
    rows, predictions = assess_profiles(config, events, truth, 731, "response")
    assert len(rows) == 28
    assert len(predictions) > 0
    assert all(row["n_test"] <= 24 for row in rows)


def test_reader_encoder_tuple_contract(tmp_path, monkeypatch):
    """The shared encoder returns (vectors, contract), not vectors alone."""
    import json
    from pathlib import Path

    from scripts import run_grain_event_evidence_study as runner

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/grain_event_evidence.json"
        ).read_text()
    )
    config["queries"] = config["queries"][:1]
    bank = runner.sentence_bank(config).head(3)

    def fake_nli(pairs):
        return pd.DataFrame(
            {
                "audit_id": pairs.audit_id,
                "nli": "neutral",
                "p_entailment": 0.1,
                "p_neutral": 0.8,
                "p_contradiction": 0.1,
            }
        )

    def fake_encoder(texts, cache, *_):
        cache.mkdir()
        metadata = {"truncated_texts": 0}
        (cache / "contract.json").write_text(json.dumps(metadata))
        return np.ones((len(texts), 4)) / 2, metadata

    monkeypatch.setattr(runner, "infer_pairs", fake_nli)
    monkeypatch.setattr(runner, "encode_batches", fake_encoder)
    result = runner.read_bank(config, bank, tmp_path)
    assert len(result) == 3
    assert result.cosine_margin.eq(0).all()
    assert not result.owner_episode_accepted.any()
