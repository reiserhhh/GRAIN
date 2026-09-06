from __future__ import annotations

import csv
from pathlib import Path

from suica_core.meps_ai_conv import load_meps_ai_conv, parse_direct_answers_markdown, public_metadata
from suica_core.meps_layout import reference_registry, resolve_meps_experiment_layout


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_participant(root: Path, name: str = "private-subject-01") -> Path:
    folder = root / name
    folder.mkdir()
    markdown = """# export
### Q1 課題1
#### 被験者回答
~~~~text
私はまず：
first response
そして：
second response
次に：
third response
~~~~
### Q2 課題2
#### 被験者回答
~~~~text
私はまず：
fourth response
そして：
fifth response
次に：
sixth response
~~~~
### Q3 課題3
#### 被験者回答
~~~~text
私はまず：
seventh response
そして：
eighth response
次に：
ninth response
~~~~
"""
    (folder / f"{name}.md").write_text(markdown, encoding="utf-8")
    _write_csv(folder / f"{name}_questions.csv", ["q_index", "answer_chars"], [
        {"q_index": 1, "answer_chars": 42},
        {"q_index": 2, "answer_chars": 43},
        {"q_index": 3, "answer_chars": 44},
    ])
    _write_csv(folder / f"{name}_split_chat.csv", ["ts_iso", "role", "q_index", "content"], [
        {"ts_iso": "2026-01-01T00:00:00", "role": "assistant", "q_index": 0, "content": "DO_NOT_SCORE"},
        {"ts_iso": "2026-01-01T00:01:00", "role": "user", "q_index": 0, "content": "user task chat"},
    ])
    _write_csv(folder / f"{name}_free_chat.csv", ["ts_iso", "role", "content"], [
        {"ts_iso": "2026-01-01T00:02:00", "role": "assistant", "content": "ALSO_DO_NOT_SCORE"},
        {"ts_iso": "2026-01-01T00:03:00", "role": "user", "content": "user free chat"},
    ])
    _write_csv(folder / f"{name}_mood.csv", ["phase", "source_scale", "response_1_to_6"], [
        {"phase": phase, "source_scale": "PANAS_NA", "response_1_to_6": 3}
        for phase in ("baseline", "pre", "mid", "post")
    ])
    for suffix in ("events", "meps_followup", "split_chat_by_question", "summary"):
        _write_csv(folder / f"{name}_{suffix}.csv", ["placeholder"], [])
    return folder


def test_direct_answer_parser_removes_fixed_scaffold() -> None:
    answers = parse_direct_answers_markdown(
        """### Q1 課題1
#### 被験者回答
~~~~text
私はまず：
alpha
そして：
beta
次に：
gamma
~~~~
"""
    )
    assert answers[1].slots == {"first": "alpha", "then": "beta", "next": "gamma"}
    assert answers[1].text == "alpha\nbeta\ngamma"
    assert answers[1].recognized_label_count == 3


def test_loader_excludes_assistant_text_and_pseudonymizes(tmp_path: Path) -> None:
    _make_participant(tmp_path)
    views, mood, metadata = load_meps_ai_conv(tmp_path, include_mood=True)
    assert metadata["n_loaded_participants"] == 1
    assert set(views["participant_code"]) == {"P001"}
    assert "private-subject-01" not in " ".join(views["participant_code"])
    assert len(views.loc[views["view_type"].eq("meps_answer")]) == 3
    task_chat = views.loc[views["view_type"].eq("meps_ai_chat")].iloc[0]
    assert task_chat["condition"] == "meps_q1"
    assert task_chat["text"] == "user task chat"
    assert "DO_NOT_SCORE" not in task_chat["text"]
    free_chat = views.loc[views["view_type"].eq("free_ai_chat")].iloc[0]
    assert free_chat["text"] == "user free chat"
    assert "ALSO_DO_NOT_SCORE" not in free_chat["text"]
    assert set(mood["phase"]) == {"baseline", "pre", "mid", "post"}
    assert "text" not in public_metadata(views).columns


def test_default_loader_never_opens_state_or_adjacent_questionnaire_files(tmp_path, monkeypatch):
    folder = _make_participant(tmp_path)
    for suffix in ("mood", "events", "meps_followup", "split_chat_by_question", "summary"):
        (folder / f"{folder.name}_{suffix}.csv").unlink()
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        assert path.name.endswith(("_questions.csv", "_free_chat.csv", "_split_chat.csv", ".md"))
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    views, mood, metadata = load_meps_ai_conv(tmp_path)
    assert len(views) > 0 and mood.empty
    assert metadata["mood_csv_loaded"] is False
    assert len(opened) == 4


def test_complete_bundle_auto_resolves_decrypted_output_and_registers_withheld_references(tmp_path: Path) -> None:
    bundle = tmp_path / "MEPS+AI_conv_experiment"
    export_root = bundle / "decrypted_output"
    export_root.mkdir(parents=True)
    _make_participant(export_root, name="0000001")
    processed = bundle / "research2_anal" / "processed"
    processed.mkdir(parents=True)
    _write_csv(processed / "pre_questionnaire_scored.csv", ["participant_id", "pre_scale"], [
        {"participant_id": "0000001", "pre_scale": 99},
    ])
    _write_csv(processed / "post_questionnaire_scored.csv", ["participant_id", "post_scale"], [
        {"participant_id": "0000001", "post_scale": 88},
    ])
    layout = resolve_meps_experiment_layout(bundle)
    assert layout.text_export_root == export_root
    assert layout.resolution_mode == "bundle_decrypted_output"
    registry = reference_registry(layout).set_index("key")
    assert registry.loc["r2_pre_questionnaire_scored", "matched_text_export_codes"] == 1
    assert registry.loc["r2_pre_questionnaire_scored", "policy"] == "WITHHELD_NOT_FOR_TRAINING_OR_VALIDATION"
    views, _mood, metadata = load_meps_ai_conv(bundle)
    assert len(views.loc[views["view_type"].eq("meps_answer")]) == 3
    assert metadata["layout_manifest"]["text_export_relative_path"] == "decrypted_output"
