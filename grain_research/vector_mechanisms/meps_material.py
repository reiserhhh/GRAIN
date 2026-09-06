"""Text-only, speaker-explicit MEPS material for frozen mechanism validation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from datetime import datetime
from datetime import time as clock_time
from pathlib import Path

import numpy as np

from suica_core.meps_ai_conv import parse_direct_answers_lines
from suica_core.meps_layout import resolve_meps_experiment_layout

ALLOWED_SUFFIXES = ("questions", "free_chat", "split_chat")
ALLOWED_COLUMNS = {
    "questions": {
        "q_index",
        "duration_sec",
        "answer_chars",
        "ai_first_use_latency_sec",
    },
    "free_chat": {"ts_iso", "role", "content_chars", "content"},
    "split_chat": {"ts_iso", "role", "q_index", "content_chars", "content"},
}


def _read_csv(path, required):
    """Read an explicitly allowed text/structural table, never a joined analysis."""
    kind = next(
        (kind for kind in ALLOWED_SUFFIXES if path.name.endswith("_" + kind + ".csv")),
        None,
    )
    if kind is None:
        raise ValueError("CSV is outside the text-only allowlist")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        if (
            len(columns) != len(set(columns))
            or not set(columns) <= ALLOWED_COLUMNS[kind]
        ):
            raise ValueError("Unexpected or duplicate text/protocol columns")
        if not set(required).issubset(columns):
            raise ValueError("Missing required text/protocol fields")
        return list(reader), columns


def parse_turns(rows):
    """Preserve file sequence and speaker; flag rather than repair time anomalies."""
    turns, counts, previous_time, previous_basis = [], Counter(), None, None
    for sequence, row in enumerate(rows):
        role = str(row.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unknown dialogue speaker")
        text = str(row.get("content", ""))
        if not text.strip():
            counts["empty_content_rows"] += 1
            continue
        stamp = str(row.get("ts_iso", ""))
        try:
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if when.tzinfo is None:
                    counts["date_without_timezone_rows"] += 1
                    basis = "naive_date"
                    seconds = (
                        when.toordinal() * 86400
                        + when.hour * 3600
                        + when.minute * 60
                        + when.second
                        + when.microsecond / 1e6
                    )
                else:
                    basis, seconds = "absolute_iso", when.timestamp()
            except ValueError:
                when = clock_time.fromisoformat(stamp)
                if when.tzinfo is not None:
                    raise ValueError("Clock-only time with unanchored zone") from None
                counts["clock_time_only_rows"] += 1
                basis = "clock_only"
                seconds = (
                    when.hour * 3600
                    + when.minute * 60
                    + when.second
                    + when.microsecond / 1e6
                )
            if previous_time is not None:
                if basis != previous_basis:
                    counts["mixed_time_basis_rows"] += 1
                elif seconds < previous_time:
                    counts[
                        "clock_order_decreases"
                        if basis == "clock_only"
                        else "nonmonotone_timestamp_rows"
                    ] += 1
            previous_time = seconds
            previous_basis = basis
        except (TypeError, ValueError, OverflowError):
            counts["unresolved_timestamp_rows"] += 1
        reported = row.get("content_chars", "")
        if reported != "":
            try:
                length = int(reported)
                if length != len(text):
                    counts["reported_not_python_codepoints"] += 1
                if length != len(text.encode("utf-16-le")) // 2:
                    counts["reported_not_utf16_units"] += 1
            except (TypeError, ValueError):
                counts["invalid_reported_char_count"] += 1
        turns.append({"role": role, "text": text, "source_sequence": sequence})
    keys = [(t["role"], t["text"]) for t in turns]
    counts["repeated_role_text_rows"] = len(keys) - len(set(keys))
    return turns, dict(counts)


def dialogue_variants(turns):
    """Keep participant evidence separate from contextual/contamination arms."""
    return {
        "user_only": "\n".join(t["text"] for t in turns if t["role"] == "user"),
        "assistant_only": "\n".join(
            t["text"] for t in turns if t["role"] == "assistant"
        ),
        "full_dialogue": "\n\n".join(
            "[" + t["role"].upper() + "]\n" + t["text"] for t in turns
        ),
    }


def _view(code, kind, condition, variants, turns, ai_possible, audit):
    """Construct private in-memory text and safe metadata as distinct objects."""
    roles = Counter(t["role"] for t in turns)
    user = variants.get("user_only", "")
    assistant = variants.get("assistant_only", "")
    metadata = {
        "participant_code": code,
        "view_kind": kind,
        "condition": condition,
        "user_chars": len(user),
        "assistant_chars": len(assistant),
        "user_turns": roles["user"],
        "assistant_turns": roles["assistant"],
        "system_turns": roles["system"],
        "ai_assistance_possible": bool(ai_possible),
        "source_order_preserved": True,
        **audit,
    }
    return {"metadata": metadata, "variants": variants, "turns": turns}


def load_text_material(input_root):
    """Load only three allowlisted CSVs plus direct-response Markdown sections.

    Mixed Markdown containers are streamed and hashed; non-response sections
    are not retained, scored, or interpreted. No mood/scale CSV is opened.
    Source IDs/text stay in memory; callers may persist metadata locally only.
    """
    layout = resolve_meps_experiment_layout(Path(input_root))
    root = layout.text_export_root
    folders = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    views, source_records, coverage, schema = [], [], [], {}
    for index, folder in enumerate(folders, 1):
        code = f"P{index:03d}"
        files = {
            kind: folder / f"{folder.name}_{kind}.csv" for kind in ALLOWED_SUFFIXES
        }
        markdown = folder / (folder.name + ".md")
        if not all(p.is_file() for p in [*files.values(), markdown]):
            raise ValueError("Required text export is missing")
        questions, columns = _read_csv(files["questions"], ["q_index", "answer_chars"])
        schema.setdefault("questions", set()).add(tuple(columns))
        indices = [int(row["q_index"]) for row in questions]
        if sorted(indices) != [1, 2, 3]:
            raise ValueError("Direct question indices are not exactly 1, 2, 3")
        with markdown.open(encoding="utf-8", errors="strict") as handle:
            answers = parse_direct_answers_lines(handle)
        if set(answers) != {1, 2, 3} or any(not a.text for a in answers.values()):
            raise ValueError("Incomplete direct-response reconstruction")
        chats = {}
        for kind in ("split_chat", "free_chat"):
            required = ["role", "content", "ts_iso"] + (
                ["q_index"] if kind == "split_chat" else []
            )
            rows, columns = _read_csv(files[kind], required)
            schema.setdefault(kind, set()).add(tuple(columns))
            if kind == "split_chat":
                if any(row["q_index"] not in {"0", "1", "2"} for row in rows):
                    raise ValueError("Assistance question index outside 0..2")
                groups = [
                    (f"meps_q{q + 1}", [r for r in rows if r["q_index"] == str(q)])
                    for q in range(3)
                ]
            else:
                groups = [("free_chat", rows)]
            for condition, rows in groups:
                turns, audit = parse_turns(rows)
                variants = dialogue_variants(turns)
                has_user = bool(variants["user_only"].strip())
                chats[condition] = {"turns": turns, "has_user": has_user}
                coverage.append(
                    {
                        "participant_code": code,
                        "condition": condition,
                        "has_user": has_user,
                        "rows": len(rows),
                        **audit,
                    }
                )
                if has_user:
                    views.append(
                        _view(
                            code, kind, condition, variants, turns, bool(turns), audit
                        )
                    )
        for q, answer in sorted(answers.items()):
            condition = f"meps_q{q}"
            variants = {"user_only": answer.text}
            turns = [{"role": "user", "text": answer.text, "source_sequence": 0}]
            views.append(
                _view(
                    code,
                    "direct_answer",
                    condition,
                    variants,
                    turns,
                    chats[condition]["has_user"],
                    {
                        "recognized_scaffold_labels": answer.recognized_label_count,
                        "slots": len(answer.slots),
                    },
                )
            )
        for kind, path in [*files.items(), ("mixed_markdown", markdown)]:
            source_records.append(
                {
                    "participant_code": code,
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return views, {
        "participants": len(folders),
        "source_files": source_records,
        "schema": {
            k: [list(v) for v in sorted(values)] for k, values in schema.items()
        },
        "chat_coverage": coverage,
        "access_policy": {
            "mood_or_questionnaire_csv_opened": False,
            "adjacent_registry_tables_opened": False,
            "mixed_markdown_container_scanned_and_hashed": True,
            "non_response_markdown_sections_retained": False,
            "assistant_text_is_context_or_negative_control_only": True,
            "model_training": False,
            "raw_text_or_source_ids_persisted": False,
        },
    }


def token_audit(views, tokenizer, *, model_name, prefix, max_tokens=512):
    """Count complete-view tokens and unknowns without truncation or embedding."""
    rows = []
    for view in views:
        for arm, text in view["variants"].items():
            if not text.strip():
                continue
            ids = tokenizer(
                prefix + text, truncation=False, add_special_tokens=True, verbose=False
            )["input_ids"]
            specials = set(tokenizer.all_special_ids)
            unk = tokenizer.unk_token_id
            unknowns = sum(i == unk for i in ids) if unk is not None else 0
            content_count = sum(i not in specials or i == unk for i in ids)
            rows.append(
                {
                    **{
                        k: view["metadata"][k]
                        for k in ("participant_code", "view_kind", "condition")
                    },
                    "model": model_name,
                    "arm": arm,
                    "chars": len(text),
                    "tokens": len(ids),
                    "unknown_tokens": unknowns,
                    "content_tokens": content_count,
                    "unknown_fraction": unknowns / content_count
                    if content_count
                    else 0,
                    "within_complete_view_budget": len(ids) <= max_tokens,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "token_sequence_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
    return rows


def split_complete_turn(text, tokenizer, *, prefix, max_tokens=512):
    """Partition one speaker's turn into complete bounded spans, without deletion.

    Bisection finds a conservative feasible prefix, not a mathematically longest
    one: token counts need not be monotone in character count. Every selected
    span is retokenized and checked. Splitting changes the context operator;
    it is not equivalent to observing the entire dialogue in one forward pass.
    """
    if not text or max_tokens < 4:
        raise ValueError("Need nonempty text and a supported token budget")

    def count(piece):
        return len(
            tokenizer(
                prefix + piece, truncation=False, add_special_tokens=True, verbose=False
            )["input_ids"]
        )

    if count("") >= max_tokens:
        raise ValueError("Prefix alone exhausts the token budget")
    start, spans = 0, []
    while start < len(text):
        end = len(text)
        if count(text[start:end]) > max_tokens:
            low, high = 0, len(text) - start
            while low + 1 < high:
                mid = (low + high) // 2
                if count(text[start : start + mid]) <= max_tokens:
                    low = mid
                else:
                    high = mid
            if low == 0:
                raise ValueError("Cannot encode one complete character in this budget")
            end = start + low
        tokens = count(text[start:end])
        if end <= start or tokens > max_tokens:
            raise ValueError("Invalid complete turn partition")
        spans.append(
            {"start": start, "end": end, "text": text[start:end], "tokens": tokens}
        )
        start = end
    if "".join(s["text"] for s in spans) != text:
        raise ValueError("Turn partition lost source characters")
    return spans
