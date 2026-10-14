#!/usr/bin/env python3
"""Verify F14 artifacts and independently recount X event/collection metadata.

This audits recorded arithmetic/provenance and X metadata via pandas rather than
the inventory's SQLite implementation. It does not independently rescan PANDORA
or treat cached-vector replay as a new psychometric validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file


def validate_aggregate_counts(profile: dict, source_rows: int) -> None:
    """Check partitions and multiplicity identities without calling them truth."""
    rows = profile["rows"]
    if rows != source_rows:
        raise ValueError("source inventory row total mismatch")
    if rows != sum(
        profile[k]
        for k in ("events", "duplicate_event_extra_rows", "missing_event_rows")
    ):
        raise ValueError("event multiplicity identity failed")
    if rows != sum(
        profile[k] for k in ("calendar_rows", "invalid_time_rows", "missing_time_rows")
    ):
        raise ValueError("time partition identity failed")
    for key in ("role_rows", "language_rows"):
        if sum(profile[key].values()) != rows:
            raise ValueError(f"{key} partition identity failed")


def recount_x(paths: list[Path]) -> dict:
    """Recount only declared X metadata, preserving event-to-tag many-to-many."""
    columns = ["post_id", "account_id", "timestamp", "symbol", "query_group", "lang"]
    frame = pd.concat(
        [
            pd.read_csv(p, usecols=columns, dtype="string", keep_default_na=False)
            for p in paths
        ],
        ignore_index=True,
    )
    for column in columns:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    complete = frame[["symbol", "query_group"]].notna().all(axis=1)
    pairs = frame.loc[complete, ["symbol", "query_group"]].drop_duplicates()
    tags = frame.loc[complete, ["post_id", "symbol", "query_group"]].drop_duplicates()
    event_groups = frame.groupby("post_id", sort=False)
    times = pd.to_datetime(frame.timestamp, format="mixed", utc=True, errors="coerce")
    dated = frame.assign(day=times.dt.strftime("%Y-%m-%d"))
    authors_days = dated.groupby("account_id").day.nunique()
    return {
        "files": len(paths),
        "rows": len(frame),
        "authors": int(frame.account_id.nunique()),
        "events": int(frame.post_id.nunique()),
        "duplicate_event_extra_rows": int(frame.post_id.duplicated().sum()),
        "repeated_event_keys": int(event_groups.size().gt(1).sum()),
        "event_author_conflicts": int(event_groups.account_id.nunique().gt(1).sum()),
        "event_raw_timestamp_conflicts": int(
            event_groups.timestamp.nunique().gt(1).sum()
        ),
        "event_language_conflicts": int(event_groups.lang.nunique().gt(1).sum()),
        "missing_author_rows": int(frame.account_id.isna().sum()),
        "missing_event_rows": int(frame.post_id.isna().sum()),
        "missing_symbol_rows": int(frame.symbol.isna().sum()),
        "missing_query_group_rows": int(frame.query_group.isna().sum()),
        "missing_condition_rows": int((~complete).sum()),
        "conditions": len(pairs),
        "event_condition_pairs": len(tags),
        "events_multiple_conditions": int(tags.groupby("post_id").size().gt(1).sum()),
        "events_without_any_complete_condition": int(
            frame.post_id.nunique() - tags.post_id.nunique()
        ),
        "authors_multiple_days": int(authors_days.gt(1).sum()),
        "same_author_condition_multiple_days": int(
            dated.loc[complete]
            .groupby(["account_id", "symbol", "query_group"])
            .day.nunique()
            .gt(1)
            .sum()
        ),
        "authors_multiple_languages": int(
            frame.groupby("account_id").lang.nunique().gt(1).sum()
        ),
        "all_timestamps_parse": bool(times.notna().all()),
        "calendar_range": [
            times.min().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            times.max().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        ],
        "exposure_denominator": "unobserved; collection tags are not response opportunities",
    }


def main() -> None:
    """Save aggregate verification in a new directory, leaving r2 untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.metadata / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or sha256_file(args.metadata / name) != expected:
            raise ValueError(f"output artifact changed: {name}")
    protocol = json.loads((args.metadata / "protocol_before_read.json").read_text())
    for path, expected in protocol["code"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"running code changed: {path}")
    summary = json.loads((args.metadata / "summary.json").read_text())
    inventory = pd.read_csv(
        args.metadata / "data_source_inventory.csv", keep_default_na=False
    )
    verified = []
    for row in inventory.itertuples(index=False):
        path = Path(row.source_path)
        if path.stat().st_size != row.bytes or sha256_file(path) != row.hash:
            raise ValueError("original source hash changed")
        verified.append(row.hash)
    for dataset, group in inventory.groupby("dataset"):
        validate_aggregate_counts(summary["datasets"][dataset], int(group.rows.sum()))
    if (
        len(inventory) != summary["input_files"]
        or int(inventory.bytes.sum()) != summary["source_bytes"]
    ):
        raise ValueError("source totals mismatch")
    x_paths = [
        Path(p) for p in inventory.loc[inventory.dataset.eq("x_market"), "source_path"]
    ]
    x = recount_x(x_paths)
    expected = summary["datasets"]["x_market"]
    keys = sorted(set(x).intersection(expected))
    for key in keys:
        if x[key] != expected[key]:
            raise ValueError(f"independent X recount mismatch: {key}")
    reference_path = args.reference / "summary.json"
    reference = json.loads(reference_path.read_text())
    for path, expected_hash in reference["input_hashes"].items():
        if sha256_file(Path(path)) != expected_hash:
            raise ValueError("reference source hash changed")
    result = {
        "status": "F14_METADATA_OUTPUTS_AND_X_RECOUNT_VERIFIED",
        "scope": "artifact/source hashes and aggregate identities for all four; independent pandas X metadata recount; not an independent PANDORA row recount",
        "metadata_manifest_sha256": sha256_file(manifest_path),
        "reference_summary_sha256": sha256_file(reference_path),
        "source_files_hashed": len(verified),
        "source_bytes": int(inventory.bytes.sum()),
        "X_independent_equal_keys": keys,
        "X_details": x,
        "new_neural_calls": 0,
        "MEPS_raw_text_or_questionnaire_read": False,
        "parent_promotion": False,
        "audit_code_sha256": sha256_file(Path(__file__)),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "source_files_hashed", "source_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
