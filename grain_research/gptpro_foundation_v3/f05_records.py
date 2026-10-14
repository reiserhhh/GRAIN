"""Declared lossless adapter from F05 scalar simulation records to F00 records."""

from __future__ import annotations


def to_common(row):
    """Represent scheduled numeric observations without inventing human text."""
    allowed = {"author_id", "role", "world", "event_id", "episode_id", "t", "z", "y"}
    if set(row) != allowed:
        raise ValueError("unexpected legacy F05 record fields")
    return {
        "schema_version": "grain-f05-common-record-v1",
        "author_id": row["author_id"],
        "event_id": row["event_id"],
        "episode_id": row["episode_id"],
        "t": row["t"],
        "condition": {"z": row["z"]},
        "opportunity": True,
        "opportunity_status": "scheduled_observed",
        "selected": True,
        "observation": row["y"],
        "text_ref": "synthetic:no_text/" + row["event_id"],
        "text_ref_kind": "synthetic_numeric_event_not_a_text_file",
        "reader_id": "f05_gaussian_simulator",
        "role": row["role"],
        "world_id": row["world"],
        "absence_reasons": {"text": "numeric simulation has no rendered text"},
    }


def from_common(row):
    """Explicit legacy-estimator view; identifiers are not predictor features."""
    if row.get("schema_version") != "grain-f05-common-record-v1" or set(
        row["condition"]
    ) != {"z"}:
        raise ValueError("unrecognized common record")
    if row["reader_id"] != "f05_gaussian_simulator" or not row["selected"]:
        raise ValueError("not a scheduled F05 simulation observation")
    return {
        "author_id": row["author_id"],
        "role": row["role"],
        "world": row["world_id"],
        "event_id": row["event_id"],
        "episode_id": row["episode_id"],
        "t": row["t"],
        "z": row["condition"]["z"],
        "y": row["observation"],
    }
