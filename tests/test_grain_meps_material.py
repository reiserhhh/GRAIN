"""Synthetic MEPS access, speaker, order, and token-budget boundary tests."""

import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest

from grain_research.vector_mechanisms.meps_material import (
    dialogue_variants,
    load_text_material,
    parse_turns,
    split_complete_turn,
    token_audit,
)
from scripts.audit_grain_meps_material import verify_outputs


def _csv(path, fields, rows):
    """Write synthetic fixture data only."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bundle(root):
    """Build text sources plus forbidden canary files."""
    d = root / "decrypted_output" / "source-private-id"
    d.mkdir(parents=True)
    sections = ["# Report", "## Mood summary", "FORBIDDEN_SCALE=999"]
    for q in range(1, 4):
        sections += [
            f"### Q{q} 課題{q}",
            "#### 被験者回答",
            "~~~text",
            "私はまず：",
            f"response {q} a",
            "そして：",
            "response b",
            "次に：",
            "response c",
            "~~~",
        ]
    (d / (d.name + ".md")).write_text("\n".join(sections))
    _csv(
        d / (d.name + "_questions.csv"),
        ["q_index", "answer_chars"],
        [{"q_index": i, "answer_chars": 42} for i in (1, 2, 3)],
    )
    rows = [
        {
            "role": "assistant",
            "ts_iso": "2026-01-01T00:00:00Z",
            "content": "AI_ONLY_CANARY",
        },
        {
            "role": "user",
            "ts_iso": "2026-01-01T00:01:00Z",
            "content": "USER_ONLY_CANARY",
        },
    ]
    _csv(d / (d.name + "_free_chat.csv"), ["role", "ts_iso", "content"], rows)
    _csv(
        d / (d.name + "_split_chat.csv"),
        ["role", "ts_iso", "content", "q_index"],
        [r | {"q_index": 0} for r in rows],
    )
    (d / (d.name + "_mood.csv")).write_text("DO_NOT_OPEN\n999")
    blocked = root / "research2_anal" / "processed"
    blocked.mkdir(parents=True)
    (blocked / "pre_questionnaire_scored.csv").write_text("DO_NOT_OPEN\n999")
    return d


def test_text_allowlist_role_separation_and_mixed_container_disclosure(
    tmp_path, monkeypatch
):
    folder = _bundle(tmp_path)
    opened = []
    original = Path.open

    def guard(path, *args, **kwargs):
        if path.is_relative_to(tmp_path):
            assert path.parent == folder
            assert path.name.endswith(
                (".md", "_questions.csv", "_free_chat.csv", "_split_chat.csv")
            )
            opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    views, audit = load_text_material(tmp_path)
    assert len(views) == 5 and audit["participants"] == 1
    assert all(v["metadata"]["participant_code"] == "P001" for v in views)
    assert "FORBIDDEN_SCALE" not in repr(views)
    assert "source-private-id" not in repr(audit)
    assert audit["access_policy"]["mood_or_questionnaire_csv_opened"] is False
    assert audit["access_policy"]["mixed_markdown_container_scanned_and_hashed"] is True
    assert len(set(opened)) == 4
    chat = next(v for v in views if v["metadata"]["view_kind"] == "free_chat")
    assert chat["variants"]["user_only"] == "USER_ONLY_CANARY"
    assert chat["variants"]["assistant_only"] == "AI_ONLY_CANARY"
    assert "[ASSISTANT]" in chat["variants"]["full_dialogue"]
    direct = next(
        v
        for v in views
        if v["metadata"]["view_kind"] == "direct_answer"
        and v["metadata"]["condition"] == "meps_q1"
    )
    assert direct["metadata"]["ai_assistance_possible"] is True


def test_time_and_utf16_anomalies_are_flagged_without_reordering():
    rows = [
        {
            "role": "user",
            "content": "hello😀",
            "content_chars": "7",
            "ts_iso": "2026-01-01T00:02:00Z",
        },
        {"role": "assistant", "content": "reply", "ts_iso": "2026-01-01T00:01:00Z"},
    ]
    turns, audit = parse_turns(rows)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert audit["reported_not_python_codepoints"] == 1
    assert audit.get("reported_not_utf16_units", 0) == 0
    assert audit["nonmonotone_timestamp_rows"] == 1
    assert dialogue_variants(turns)["full_dialogue"].startswith("[USER]")
    with pytest.raises(ValueError, match="speaker"):
        parse_turns([{"role": "unknown", "content": "x"}])


def test_new_questionnaire_column_is_rejected_before_row_iteration(
    tmp_path, monkeypatch
):
    folder = _bundle(tmp_path)
    path = folder / (folder.name + "_questions.csv")
    _csv(path, ["q_index", "answer_chars", "mood_score"], [])

    def reject_iteration(_reader):
        raise AssertionError("Unexpected schema must be rejected before reading values")

    monkeypatch.setattr(csv.DictReader, "__next__", reject_iteration)
    with pytest.raises(ValueError, match="Unexpected"):
        load_text_material(tmp_path)


def test_export_clock_times_are_not_misrepresented_as_absolute_dates():
    turns, audit = parse_turns(
        [
            {"role": "user", "content": "first", "ts_iso": "23:59:59.999"},
            {"role": "assistant", "content": "second", "ts_iso": "00:00:00.001"},
        ]
    )
    assert audit["clock_time_only_rows"] == 2
    assert audit["clock_order_decreases"] == 1
    assert audit.get("unresolved_timestamp_rows", 0) == 0
    assert [t["text"] for t in turns] == ["first", "second"]


def test_complete_token_audit_never_truncates_or_persists_token_ids(tmp_path):
    _bundle(tmp_path)
    views, _ = load_text_material(tmp_path)

    class DummyTokenizer:
        all_special_ids = (100,)
        unk_token_id = 0

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [100] + [1] * len(text) + [100]}

    rows = token_audit(
        views, DummyTokenizer(), model_name="dummy", prefix="query: ", max_tokens=5
    )
    assert len(rows) == 9
    assert not any(r["within_complete_view_budget"] for r in rows)
    assert not any("input_ids" in r for r in rows)
    assert all(r["unknown_fraction"] == 0 for r in rows)


def test_turn_partition_preserves_whitespace_unicode_and_all_characters():
    class CharacterTokenizer:
        def __call__(self, text, **kwargs):
            assert not kwargs["truncation"]
            return {"input_ids": [1] + list(text) + [2]}

    text = "  one\n日本語😀\n  two  " * 10
    spans = split_complete_turn(
        text, CharacterTokenizer(), prefix="query: ", max_tokens=24
    )
    assert len(spans) > 1
    assert "".join(s["text"] for s in spans) == text
    assert all(s["tokens"] <= 24 for s in spans)
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in pairwise(spans))
    with pytest.raises(ValueError, match="Prefix"):
        split_complete_turn(text, CharacterTokenizer(), prefix="query: ", max_tokens=8)


def test_metadata_audit_rejects_changed_bytes_and_path_escape(tmp_path):
    artifact = tmp_path / "metadata.csv"
    artifact.write_text("count\n1\n")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "summary.json").write_text(
        json.dumps({"outputs_sha256": {"metadata.csv": expected}})
    )
    assert verify_outputs(tmp_path)["outputs_sha256"]["metadata.csv"] == expected
    artifact.write_text("count\n2\n")
    with pytest.raises(ValueError, match="Changed"):
        verify_outputs(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps({"outputs_sha256": {"../outside.csv": expected}})
    )
    with pytest.raises(ValueError, match="escapes"):
        verify_outputs(tmp_path)
