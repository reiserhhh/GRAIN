"""Private-data adapter for the MEPS + AI conversation protocol.

The protocol has three analytically different user-text views:

* direct answers to three shared MEPS prompts;
* optional user turns in the AI-assistance window for each prompt; and
* a free AI-conversation window.

Assistant turns are deliberately *not* part of a participant's response text.
They remain opportunity/context metadata (turn count and character count).
This matters for SUICA: importing the assistant's language as if it belonged to
the participant would create a direct measurement contamination path.

No function here writes raw text or source participant IDs.  Callers that need
to persist audit artefacts should use the metadata-only tables returned by
``public_metadata``.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from suica_core.meps_layout import resolve_meps_experiment_layout


QUESTION_HEADING_RE = re.compile(r"^Q(?P<question>[1-3])\s+課題[1-3]\s*$")
HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
ANSWER_SLOT_LABELS = {
    "私はまず：": "first",
    "そして：": "then",
    "次に：": "next",
}
EXPECTED_SUFFIXES = (
    "events",
    "free_chat",
    "meps_followup",
    "mood",
    "questions",
    "split_chat",
    "split_chat_by_question",
    "summary",
)


@dataclass(frozen=True)
class DirectAnswer:
    """One direct MEPS response reconstructed from the Markdown export.

    ``text`` excludes the fixed rhetorical scaffold ("I first", "then",
    "next") because the scaffold is an expression opportunity supplied by the
    protocol, not a participant difference.  The response slots are kept so a
    later analysis can explicitly model rhetorical position instead of silently
    treating it as a trait feature.
    """

    question_index: int
    text: str
    slots: dict[str, str]
    raw_body_character_count: int
    recognized_label_count: int


def _eligible_participant_dirs(input_root: Path) -> list[Path]:
    """Return stable participant directories while excluding Finder artefacts."""
    return sorted(
        path for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "._"))
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _clean_markdown_line(value: str) -> str:
    """Remove only export syntax; preserve all participant-authored wording."""
    return value.strip()


def parse_direct_answers_markdown(markdown: str) -> dict[int, DirectAnswer]:
    """Recover direct MEPS answer text and its fixed rhetorical slots.

    The known export format nests ``#### 被験者回答`` under each ``### Qn 課題n``
    section and surrounds content with a Markdown code fence.  The parser is
    deliberately conservative: unfamiliar non-empty lines are retained as an
    ``unlabeled`` slot rather than discarded.
    """
    return parse_direct_answers_lines(markdown.splitlines())


def parse_direct_answers_lines(lines: Iterable[str]) -> dict[int, DirectAnswer]:
    """Extract response sections while discarding unrelated mixed-export sections.

    A Markdown container can include state summaries. Iterating its lines is
    still a file read, but only direct-response sections are retained/parsed as
    content. This must not be described as never accessing the mixed container.
    """
    collected: dict[int, list[str]] = {}
    current_question: int | None = None
    collecting_question: int | None = None

    for raw_line in lines:
        raw_line = raw_line.rstrip("\r\n")
        heading = HEADING_RE.match(raw_line)
        if heading:
            level = len(heading.group("level"))
            title = heading.group("title").strip()
            question = QUESTION_HEADING_RE.match(title)
            if question:
                current_question = int(question.group("question"))
            if level <= 4:
                collecting_question = (
                    current_question
                    if level == 4 and title == "被験者回答" and current_question is not None
                    else None
                )
            continue

        if collecting_question is not None and raw_line.strip():
            collected.setdefault(collecting_question, []).append(raw_line.rstrip())

    answers: dict[int, DirectAnswer] = {}
    for question_index, raw_lines in collected.items():
        slots: dict[str, list[str]] = {}
        active_slot = "unlabeled"
        recognized_label_count = 0
        body_lines: list[str] = []
        for raw_line in raw_lines:
            line = _clean_markdown_line(raw_line)
            if not line or line.startswith(("~~~", "```")):
                continue
            if line in ANSWER_SLOT_LABELS:
                active_slot = ANSWER_SLOT_LABELS[line]
                recognized_label_count += 1
                continue
            slots.setdefault(active_slot, []).append(line)
            body_lines.append(line)

        clean_slots = {
            slot: "\n".join(lines).strip()
            for slot, lines in slots.items()
            if "\n".join(lines).strip()
        }
        answers[question_index] = DirectAnswer(
            question_index=question_index,
            text="\n".join(body_lines).strip(),
            slots=clean_slots,
            raw_body_character_count=sum(len(line) for line in raw_lines),
            recognized_label_count=recognized_label_count,
        )
    return answers


def _join_turns(rows: Iterable[dict[str, str]], role: str) -> tuple[str, int, int]:
    """Join one role's turns chronologically and return text, count, chars."""
    selected = [row for row in rows if str(row.get("role", "")).lower() == role]
    selected.sort(key=lambda row: str(row.get("ts_iso", "")))
    texts = [str(row.get("content", "")).strip() for row in selected]
    texts = [text for text in texts if text]
    return "\n".join(texts), len(texts), int(sum(len(text) for text in texts))


def _question_char_counts(rows: Iterable[dict[str, str]]) -> dict[int, int]:
    values: dict[int, int] = {}
    for row in rows:
        try:
            values[int(row["q_index"])] = int(row["answer_chars"])
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _mood_rows(rows: Iterable[dict[str, str]], participant_code: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        try:
            value = float(row["response_1_to_6"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(row.get("phase", "")), str(row.get("source_scale", "")))
        grouped.setdefault(key, []).append(value)
    return [
        {
            "participant_code": participant_code,
            "phase": phase,
            "source_scale": scale,
            "n_items": len(values),
            "mean_response": float(np.mean(values)),
        }
        for (phase, scale), values in sorted(grouped.items())
    ]


def load_meps_ai_conv(
    input_root: Path, *, include_mood: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load private MEPS text; state tables require explicit opt-in.

    ``views`` contains raw participant text *only in memory*.  It includes no
    source participant ID.  The stable ``P001`` coding is sufficient for
    within-run joins and makes metadata outputs safe to retain locally.  The
    caller must not export the ``text`` column to reports or version control.
    """
    layout = resolve_meps_experiment_layout(input_root)
    input_root = layout.text_export_root

    participants = _eligible_participant_dirs(input_root)
    if not participants:
        raise ValueError(f"no participant directories found under: {input_root}")

    view_rows: list[dict[str, Any]] = []
    mood_rows: list[dict[str, Any]] = []
    missing_files: dict[str, list[str]] = {}
    direct_answer_qc: list[dict[str, Any]] = []

    for position, folder in enumerate(participants, start=1):
        participant_code = f"P{position:03d}"
        files = {suffix: folder / f"{folder.name}_{suffix}.csv" for suffix in EXPECTED_SUFFIXES}
        required = ("questions", "free_chat", "split_chat") + (("mood",) if include_mood else ())
        missing = [suffix for suffix in required if not files[suffix].exists()]
        if missing:
            missing_files[participant_code] = missing
            continue
        markdown_path = folder / f"{folder.name}.md"
        if not markdown_path.exists():
            missing_files[participant_code] = [*missing_files.get(participant_code, []), "markdown"]
            continue

        questions = _question_char_counts(_read_csv(files["questions"]))
        with markdown_path.open(encoding="utf-8", errors="strict") as handle:
            direct_answers = parse_direct_answers_lines(handle)
        for question_index in (1, 2, 3):
            answer = direct_answers.get(question_index)
            if answer is None:
                direct_answer_qc.append({
                    "participant_code": participant_code,
                    "question_index": question_index,
                    "recovered": False,
                    "recognized_label_count": 0,
                    "reported_answer_chars": questions.get(question_index),
                    "recovered_answer_chars": 0,
                    "reported_minus_recovered": np.nan,
                })
                continue
            reported_chars = questions.get(question_index)
            recovered_chars = len(answer.text)
            direct_answer_qc.append({
                "participant_code": participant_code,
                "question_index": question_index,
                "recovered": bool(answer.text),
                "recognized_label_count": answer.recognized_label_count,
                "reported_answer_chars": reported_chars,
                "recovered_answer_chars": recovered_chars,
                "reported_minus_recovered": (
                    float(reported_chars - recovered_chars) if reported_chars is not None else np.nan
                ),
            })
            if answer.text:
                view_rows.append({
                    "participant_code": participant_code,
                    "view_type": "meps_answer",
                    "condition": f"meps_q{question_index}",
                    "question_index": question_index,
                    "slot": "aggregate",
                    "text": answer.text,
                    "user_turns": 1,
                    "user_chars": recovered_chars,
                    "assistant_turns": 0,
                    "assistant_chars": 0,
                    "reported_answer_chars": reported_chars,
                })
            for slot, slot_text in answer.slots.items():
                view_rows.append({
                    "participant_code": participant_code,
                    "view_type": "meps_answer_slot",
                    "condition": f"meps_q{question_index}",
                    "question_index": question_index,
                    "slot": slot,
                    "text": slot_text,
                    "user_turns": 1,
                    "user_chars": len(slot_text),
                    "assistant_turns": 0,
                    "assistant_chars": 0,
                    "reported_answer_chars": reported_chars,
                })

        split_rows = _read_csv(files["split_chat"])
        for raw_question_index in (0, 1, 2):
            conversation_rows = [
                row for row in split_rows
                if str(row.get("q_index", "")) == str(raw_question_index)
            ]
            user_text, user_turns, user_chars = _join_turns(conversation_rows, "user")
            _assistant_text, assistant_turns, assistant_chars = _join_turns(conversation_rows, "assistant")
            if user_text:
                view_rows.append({
                    "participant_code": participant_code,
                    "view_type": "meps_ai_chat",
                    "condition": f"meps_q{raw_question_index + 1}",
                    "question_index": raw_question_index + 1,
                    "slot": "aggregate",
                    "text": user_text,
                    "user_turns": user_turns,
                    "user_chars": user_chars,
                    "assistant_turns": assistant_turns,
                    "assistant_chars": assistant_chars,
                    "reported_answer_chars": questions.get(raw_question_index + 1),
                })

        free_rows = _read_csv(files["free_chat"])
        user_text, user_turns, user_chars = _join_turns(free_rows, "user")
        _assistant_text, assistant_turns, assistant_chars = _join_turns(free_rows, "assistant")
        if user_text:
            view_rows.append({
                "participant_code": participant_code,
                "view_type": "free_ai_chat",
                "condition": "free_chat",
                "question_index": np.nan,
                "slot": "aggregate",
                "text": user_text,
                "user_turns": user_turns,
                "user_chars": user_chars,
                "assistant_turns": assistant_turns,
                "assistant_chars": assistant_chars,
                "reported_answer_chars": np.nan,
            })

        if include_mood:
            mood_rows.extend(_mood_rows(_read_csv(files["mood"]), participant_code))

    views = pd.DataFrame(view_rows)
    mood = pd.DataFrame(mood_rows, columns=[
        "participant_code", "phase", "source_scale", "n_items", "mean_response",
    ])
    qc = pd.DataFrame(direct_answer_qc)
    metadata: dict[str, Any] = {
        "layout_manifest": layout.public_manifest(),
        "n_discovered_participants": len(participants),
        "n_loaded_participants": len(participants) - len(missing_files),
        "mood_csv_loaded": bool(include_mood),
        "mixed_markdown_container_scanned": True,
        "non_response_markdown_sections_retained": False,
        "missing_files": missing_files,
        "direct_answer_qc": qc,
        "source_schema": {
            "text_exports": "auto-resolved decrypted_output/ when a complete experiment bundle is supplied",
            "direct_answers": "Markdown: Qn section -> 被験者回答 block",
            "task_assistance": "split_chat.csv q_index 0..2 mapped to MEPS q_index 1..3",
            "free_conversation": "free_chat.csv",
            "mood": "mood.csv phase x source_scale",
        },
    }
    return views, mood, metadata


def public_metadata(views: pd.DataFrame) -> pd.DataFrame:
    """Drop raw text before persisting a table or including it in a report."""
    return views.drop(columns=["text"], errors="ignore").copy()
