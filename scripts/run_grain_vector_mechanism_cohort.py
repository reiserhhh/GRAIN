#!/usr/bin/env python3
"""Stream a new, label-blind PANDORA cohort with three 12-comment views each."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import (
    COHORT_SEED,
    ID_SEED,
    select_multicomment_bundle,
    summarize_bundles,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import (
    leakage_flags,
    normalized_text_key,
    stable_key,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Select IDs before reading comments; split eligible authors before analysis."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--exclude-views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-authors", type=int, default=1024)
    p.add_argument("--reservoir", type=int, default=192)
    p.add_argument("--units", type=int, default=12)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=100)
    a = p.parse_args()
    if a.candidate_authors < 240 or a.reservoir < 3 * a.units or a.units < 12:
        raise ValueError(
            "Need >=240 candidates, >=12 units/view and sufficient reservoir"
        )
    if not 1 <= a.min_words <= a.max_words:
        raise ValueError("Invalid word budget")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new empty cohort directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    start = time.monotonic()
    excluded_source = json.loads(a.exclude_views.read_text(encoding="utf-8"))
    excluded = {b["author_id"] for b in excluded_source}
    with (a.data_root / "PANDORA_official/author_profiles.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        if "author" not in reader.fieldnames:
            raise ValueError("Missing author IDs")
        authors = {r["author"].strip() for r in reader if r["author"].strip()}
    candidates = sorted(
        (u for u in authors if stable_key(u, ID_SEED)[:24] not in excluded),
        key=lambda u: stable_key(u, COHORT_SEED),
    )[: a.candidate_authors]
    heaps = {u: [] for u in candidates}
    ids = {u: set() for u in candidates}
    counts = Counter()
    csv.field_size_limit(10_000_000)
    path = a.data_root / "PANDORA_official/all_comments_since_2015.csv"
    before = path.stat()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {"author", "body", "lang", "created_utc", "id", "link_id", "subreddit"}
        if not fields.issubset(reader.fieldnames):
            raise ValueError("Comment schema missing required fields")
        for r in reader:
            counts["rows_scanned"] += 1
            if counts["rows_scanned"] % 1_000_000 == 0:
                print(
                    f"SCANNED {counts['rows_scanned']} seconds={time.monotonic() - start:.1f}",
                    flush=True,
                )
            user = r["author"]
            if user not in heaps:
                continue
            counts["candidate_comments"] += 1
            if r["lang"] != "en":
                counts["non_english"] += 1
                continue
            n = len(r["body"].split())
            if not a.min_words <= n <= a.max_words:
                counts["word_budget_excluded"] += 1
                continue
            if not r["id"] or not r["link_id"] or r["id"] in ids[user]:
                counts["missing_or_repeated_id_thread"] += 1
                continue
            try:
                t = float(r["created_utc"])
            except ValueError:
                t = float("nan")
            if not 1_000_000_000 <= t < 3_000_000_000:
                counts["invalid_time"] += 1
                continue
            counts["eligible_comment_rows"] += 1
            rank = int(stable_key(r["id"], COHORT_SEED), 16)
            heap = heaps[user]
            if len(heap) == a.reservoir and rank >= -heap[0][0]:
                continue
            item = {
                "author_id": stable_key(user, ID_SEED)[:24],
                "comment_id": stable_key(r["id"], ID_SEED)[:24],
                "thread_id": stable_key(r["link_id"], ID_SEED)[:24],
                "body": r["body"],
                "timestamp": t,
                "word_count": n,
                "subreddit": r["subreddit"],
                "leakage_flags": leakage_flags(r["body"]),
            }
            entry = (-rank, r["id"], item)
            if len(heap) == a.reservoir:
                old = heapq.heapreplace(heap, entry)
                ids[user].remove(old[1])
            else:
                heapq.heappush(heap, entry)
            ids[user].add(r["id"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw source changed during scan")
    text_authors = defaultdict(set)
    for heap in heaps.values():
        for _, _, c in heap:
            text_authors[normalized_text_key(c["body"])].add(c["author_id"])
    forbidden = {k for k, users in text_authors.items() if len(users) > 1}
    bundles = []
    for user in candidates:
        try:
            b = select_multicomment_bundle(
                [c for _, _, c in heaps[user]], units=a.units, forbidden_texts=forbidden
            )
        except ValueError as exc:
            if "temporal separation" not in str(exc):
                raise
            counts["tied_period_boundary_authors"] += 1
            b = None
        if b:
            bundles.append(b)
    selected = bundles[:240]
    for i, b in enumerate(selected):
        b["split"] = (
            "discovery"
            if i < 144
            else "calibration"
            if i < 192
            else "mechanism_holdout"
        )
        b["discovery_role"] = (
            "reference" if i < 48 else "comparison" if i < 144 else "unopened"
        )
    if not selected:
        raise ValueError("No eligible authors")
    assert not ({b["author_id"] for b in selected} & excluded)
    write_json(a.output_dir / "author_views.json", selected)
    # Analysis runners open only the requested split file, never the holdout by default.
    for split in ("discovery", "calibration", "mechanism_holdout"):
        write_json(
            a.output_dir / f"{split}_views.json",
            [b for b in selected if b["split"] == split],
        )
    audit = {
        **dict(counts),
        **summarize_bundles(selected),
        "status": "COMPLETE" if len(selected) == 240 else "INSUFFICIENT_MATERIAL",
        "candidate_authors": len(candidates),
        "eligible_authors": len(bundles),
        "excluded_previous_authors": len(excluded),
        "overlap_previous_authors": 0,
        "normalized_cross_author_duplicate_bodies_removed": len(forbidden),
        "duplicate_scope": "retained candidate reservoirs only; semantic near-duplicates not cleared",
        "view_sha256": material_hash(selected),
        "exclude_view_sha256": material_hash(excluded_source),
        "selection_seed": COHORT_SEED,
        "id_seed": ID_SEED,
        "units_per_view": a.units,
        "reservoir": a.reservoir,
        "min_words": a.min_words,
        "max_words": a.max_words,
        "raw_source": path.name,
        "source_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "labels_used": False,
        "code_sha256": code_fingerprint(ROOT),
        "exposure": "New experiment authors; historically investigated PANDORA corpus and frozen GRAIN",
        "selection_caveat": f"English {a.min_words}-{a.max_words} word comments, active multi-thread authors; not all PANDORA",
        "seconds": time.monotonic() - start,
    }
    write_json(a.output_dir / "data_inventory.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
