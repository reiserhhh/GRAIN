#!/usr/bin/env python3
"""Event identity and collection context adapters for existing text observations.

Published authorship is not the subject/actuality of every clause. These adapters
restore material provenance; they do not infer semantic evidence qualification.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def text_hash(value: str) -> str:
    """Hash exact UTF-8 text without cleaning or normalizing its content."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event_address_frame(material, cached_authors, addresses, unit_authors, conditions):
    """Restore comment identity only after exact author, body and span alignment."""
    bundles, documents = material["bundles"], material["documents"]
    ids = [b["author_id"] for b in bundles]
    if ids != cached_authors["ids"] or len(set(ids)) != len(ids):
        raise ValueError("material author order/uniqueness mismatch")
    if len(documents) != len(bundles):
        raise ValueError("document count mismatch")
    if [text_hash(d) for d in documents] != cached_authors["document_hashes"]:
        raise ValueError("document content hash mismatch")
    rows = []
    for author_index, bundle in enumerate(bundles):
        if bundle["source_group"] != "big5_only":
            raise ValueError("unexpected source population")
        comments = [c for view in bundle["views"] for c in view]
        if "\n\n".join(c["body"] for c in comments) != documents[author_index]:
            raise ValueError("comment aggregation differs from cached document")
        start = 0
        unit = 0
        previous_time = -np.inf
        for view_index, view in enumerate(bundle["views"]):
            for comment in view:
                if comment["author_id"] != bundle["author_id"]:
                    raise ValueError("comment author mismatch")
                timestamp = float(comment["timestamp"])
                if not np.isfinite(timestamp) or timestamp < previous_time:
                    raise ValueError("invalid or reversed calendar time")
                previous_time = timestamp
                end = start + len(comment["body"])
                rows.append(
                    {
                        "author_index": author_index,
                        "unit": unit,
                        "start": start,
                        "end": end,
                        "author_key": bundle["author_id"],
                        "event_key": comment["comment_id"],
                        "thread_key": comment["thread_id"],
                        "condition": comment["subreddit"],
                        "timestamp_unix_seconds": timestamp,
                        "view_index": view_index,
                        "body_sha256": text_hash(comment["body"]),
                        "authorship_scope": "published_comment_not_semantic_self_evidence",
                    }
                )
                start, unit = end + 2, unit + 1
    out = pd.DataFrame(rows)
    keys = ["author_index", "unit", "start", "end"]
    if len(out) != len(addresses) or not np.array_equal(
        out[keys].to_numpy(), addresses[keys].to_numpy()
    ):
        raise ValueError("cached material span mismatch")
    if not np.array_equal(out.author_index, unit_authors) or not np.array_equal(
        out.condition, conditions
    ):
        raise ValueError("cached vector author/condition order mismatch")
    if (
        out.event_key.isna().any()
        or out.event_key.eq("").any()
        or out.event_key.duplicated().any()
    ):
        raise ValueError("missing or repeated comment event")
    if out.duplicated(["author_key", "thread_key"]).any():
        raise ValueError("source bundle unexpectedly repeats a discussion")
    out["time_utc"] = pd.to_datetime(
        out.timestamp_unix_seconds, unit="s", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return out


def verify_event_matches(expected: pd.DataFrame, found: pd.DataFrame) -> dict:
    """Compare raw-source metadata to cached events, retaining repeat counts."""
    if expected.event_key.duplicated().any() or len(expected) == 0:
        raise ValueError("unique nonempty expected events required")
    keys = ["author_key", "thread_key", "condition", "timestamp_unix_seconds"]
    if found[["event_key", *keys]].isna().any().any():
        raise ValueError("missing matched source metadata")
    if set(expected.event_key) != set(found.event_key):
        raise ValueError("missing or extraneous raw event matches")
    for key in keys:
        if found.groupby("event_key")[key].nunique().gt(1).any():
            raise ValueError(f"conflicting duplicate raw event: {key}")
    source = found.drop_duplicates("event_key").set_index("event_key")
    e = expected.set_index("event_key")
    for key in keys:
        if not np.array_equal(e[key], source.loc[e.index, key]):
            raise ValueError(f"source/cache event mismatch: {key}")
    return {
        "matched_unique_events": len(expected),
        "matched_source_rows": len(found),
        "repeat_source_rows": len(found) - len(expected),
        "all_event_author_time_thread_condition_match": True,
        "raw_body_compared": False,
    }


def x_event_tables(frame: pd.DataFrame):
    """Separate one post from its possibly multiple/partial collection contexts.

    Raw IDs remain internal to the caller. Timestamps are parsed explicitly with
    a timezone; contradictory identity, time or language is rejected rather than
    silently resolved by keeping the first row. Tags may legitimately differ.
    """
    columns = ["post_id", "account_id", "timestamp", "lang", "symbol", "query_group"]
    f = frame[columns].copy()
    for col in columns:
        f[col] = f[col].astype("string").str.strip().replace("", pd.NA)
    if f[["post_id", "account_id", "timestamp"]].isna().any().any():
        raise ValueError("post, author and timestamp are required")
    tz = f.timestamp.str.contains(
        r"^\d{4}-\d\d-\d\d[T ].*(?:Z|[+-]\d\d:\d\d)$", na=False
    )
    times = pd.to_datetime(
        f.timestamp.where(tz), format="mixed", utc=True, errors="coerce"
    )
    if times.isna().any():
        raise ValueError("explicit calendar date and timezone required")
    f["time_utc"] = times.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for column in ("account_id", "time_utc", "lang"):
        if f.groupby("post_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"conflicting post metadata: {column}")
    events = f[["post_id", "account_id", "time_utc", "lang"]].drop_duplicates("post_id")
    counts = f.groupby("post_id").size()
    events["collection_rows"] = events.post_id.map(counts)
    tags = f[["post_id", "symbol", "query_group"]].drop_duplicates().copy()
    tags["complete_condition"] = tags[["symbol", "query_group"]].notna().all(axis=1)
    tags["condition_key"] = tags.apply(
        lambda r: (
            json.dumps([str(r.symbol), str(r.query_group)])
            if r.complete_condition
            else None
        ),
        axis=1,
    )
    tags["partial_reason"] = tags.apply(
        lambda r: "+".join(k for k in ("symbol", "query_group") if pd.isna(r[k])),
        axis=1,
    )
    events = events.sort_values(
        ["account_id", "time_utc", "post_id"], kind="stable"
    ).reset_index(drop=True)
    return events, tags.reset_index(drop=True)
