#!/usr/bin/env python3
"""Build private VM02 pilot views by streaming official raw comments once."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import (
    leakage_flags,
    select_context_bundle,
    stable_key,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Sample by ID hash, never by Big5/MBTI values; keep raw bodies private."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-authors", type=int, default=128)
    parser.add_argument("--reservoir", type=int, default=64)
    parser.add_argument("--min-words", type=int, default=60)
    parser.add_argument("--max-words", type=int, default=180)
    args = parser.parse_args()
    if not 5 <= args.reservoir or args.candidate_authors < 32:
        raise ValueError(
            "Require >=32 author candidates and >=5 comments per reservoir"
        )
    seed = "grain_vm02_raw_contexts_20260905"
    csv.field_size_limit(10_000_000)
    profiles = args.data_root / "PANDORA_official/author_profiles.csv"
    comments = args.data_root / "PANDORA_official/all_comments_since_2015.csv"
    with profiles.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "author" not in (reader.fieldnames or []):
            raise ValueError("Profiles lack author IDs")
        authors = {r["author"].strip() for r in reader if r["author"].strip()}
    candidates = sorted(authors, key=lambda u: stable_key(u, seed))[
        : args.candidate_authors
    ]
    heaps = {u: [] for u in candidates}
    counts, retained_ids = Counter(), {u: set() for u in candidates}
    start = time.monotonic()
    with comments.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "author",
            "body",
            "created_utc",
            "subreddit",
            "link_id",
            "id",
            "lang",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing fields: {sorted(required - set(reader.fieldnames or []))}"
            )
        for row in reader:
            counts["rows_scanned"] += 1
            if counts["rows_scanned"] % 500000 == 0:
                print(
                    f"SCANNED {counts['rows_scanned']} elapsed={time.monotonic() - start:.1f}",
                    flush=True,
                )
            user = row["author"]
            if user not in heaps:
                continue
            counts["candidate_rows"] += 1
            if row["lang"] != "en":
                counts["non_english"] += 1
                continue
            words = len(row["body"].split())
            if not args.min_words <= words <= args.max_words:
                counts["word_budget_excluded"] += 1
                continue
            if not row["id"] or not row["link_id"] or row["id"] in retained_ids[user]:
                counts["id_thread_missing_or_duplicate"] += 1
                continue
            try:
                timestamp = float(row["created_utc"])
            except ValueError:
                counts["timestamp_invalid"] += 1
                continue
            if not 1_000_000_000 <= timestamp < 3_000_000_000:
                counts["timestamp_invalid"] += 1
                continue
            rank = int(stable_key(row["id"], seed), 16)
            item = {
                "author_id": stable_key(user, seed)[:24],
                "comment_id": stable_key(row["id"], seed)[:24],
                "thread_id": stable_key(row["link_id"], seed)[:24],
                "timestamp": timestamp,
                "subreddit": row["subreddit"],
                "body": row["body"],
                "word_count": words,
                "leakage_flags": leakage_flags(row["body"]),
            }
            entry = (-rank, row["id"], item)
            heap = heaps[user]
            if len(heap) < args.reservoir:
                heapq.heappush(heap, entry)
                retained_ids[user].add(row["id"])
            elif rank < -heap[0][0]:
                dropped = heapq.heapreplace(heap, entry)
                retained_ids[user].remove(dropped[1])
                retained_ids[user].add(row["id"])
    bundles = []
    for user in candidates:
        bundle = select_context_bundle([entry[2] for entry in heaps[user]])
        if bundle is not None:
            bundles.append(bundle)
    selected = bundles[:32]
    for index, bundle in enumerate(selected):
        bundle["split"] = "technical_only" if index < 8 else "pilot_development"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "author_views.json", selected)
    audit = {
        **dict(counts),
        "candidate_authors": len(candidates),
        "eligible_authors": len(bundles),
        "technical_authors": min(8, len(selected)),
        "pilot_authors": max(0, len(selected) - 8),
        "status": "COMPLETE" if len(selected) == 32 else "INSUFFICIENT_MATERIAL",
        "source": comments.name,
        "source_bytes": comments.stat().st_size,
        "source_mtime_ns": comments.stat().st_mtime_ns,
        "view_sha256": material_hash(selected),
        "seed": seed,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "reservoir": args.reservoir,
        "history_per_author": 3,
        "current_per_author": 2,
        "strict_temporal_and_thread_separation": True,
        "labels_used_for_selection": False,
        "history_exposure": "previously researched PANDORA, not a new confirmatory lockbox",
        "selection_caveat": "English medium-length comments and multi-thread coverage; not population representative",
        "seconds": time.monotonic() - start,
    }
    for flag in ["mbti_type_term", "big5_term", "report_clue"]:
        audit[f"{flag}_comments"] = sum(
            c["leakage_flags"][flag]
            for b in selected
            for c in b["history"] + b["current"]
        )
    write_json(args.output_dir / "data_inventory.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
