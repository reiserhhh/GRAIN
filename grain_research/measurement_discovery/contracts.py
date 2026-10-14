"""Small schema contracts for auditable contextual measurement experiments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

RECORD_FIELDS = (
    "author_id",
    "event_id",
    "episode_id",
    "t",
    "condition",
    "opportunity",
    "opportunity_status",
    "selected",
    "observation",
    "text_ref",
    "reader_id",
    "role",
    "condition_status",
)
TARGET_FIELDS = (
    "target_id",
    "response_range",
    "reference_population",
    "condition_universe",
    "observation_policy",
    "unit",
    "allowed_interpretation",
)


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_records(records: Iterable[Mapping], *, require_text_ref=True):
    """Validate record identity and nullable observation fields without fitting."""
    rows = list(records)
    seen_events = {}
    for row in rows:
        if set(row) != set(RECORD_FIELDS):
            raise ValueError("record fields do not match the discovery contract")
        for name in ("author_id", "event_id", "episode_id", "role"):
            _required_text(row[name], name)
        if require_text_ref and row["text_ref"] is not None:
            _required_text(row["text_ref"], "text_ref")
        if not isinstance(row["t"], (int, float)):
            raise ValueError("t must be numeric or null")
        if row["event_id"] in seen_events and seen_events[row["event_id"]] != row:
            raise ValueError("event_id is reused with conflicting records")
        seen_events[row["event_id"]] = row
        if row["opportunity_status"] not in {
            "observed", "unknown", "not_applicable", None
        }:
            raise ValueError("invalid opportunity_status")
        if row["condition_status"] not in {
            "assigned", "recorded", "self_selected", "unknown", None
        }:
            raise ValueError("invalid condition_status")
    return {
        "rows": len(rows),
        "authors": len({r["author_id"] for r in rows}),
        "episodes": len({r["episode_id"] for r in rows}),
        "events": len(seen_events),
        "missing_text_ref": sum(r["text_ref"] is None for r in rows),
        "opportunity_status": dict(Counter(r["opportunity_status"] for r in rows)),
    }


def validate_split(records: Iterable[Mapping], split_by_author: Mapping[str, str]):
    """Ensure authors, not individual rows, define source/validation/application."""
    rows = list(records)
    allowed = {"source", "validation", "application"}
    if not split_by_author:
        raise ValueError("author-level split is required")
    if set(split_by_author.values()) - allowed:
        raise ValueError("unknown split label")
    observed = {row["author_id"] for row in rows}
    if observed - set(split_by_author):
        raise ValueError("every record author needs a split")
    by_split = {name: set() for name in allowed}
    for author, split in split_by_author.items():
        by_split[split].add(author)
    if set.intersection(*by_split.values()):
        raise ValueError("author appears in more than one split")
    return {name: len(authors) for name, authors in by_split.items()}


def target_contract(**kwargs):
    """Create a complete target declaration with explicit interpretation scope."""
    if set(kwargs) != set(TARGET_FIELDS):
        missing = set(TARGET_FIELDS) - set(kwargs)
        extra = set(kwargs) - set(TARGET_FIELDS)
        raise ValueError(f"target fields mismatch; missing={missing}, extra={extra}")
    _required_text(kwargs["target_id"], "target_id")
    for field in TARGET_FIELDS[1:]:
        value = kwargs[field]
        if isinstance(value, str):
            _required_text(value, field)
        elif not value:
            raise ValueError(f"{field} cannot be empty")
    return dict(kwargs)
