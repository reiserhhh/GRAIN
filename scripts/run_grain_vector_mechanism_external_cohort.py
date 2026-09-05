#!/usr/bin/env python3
"""Stage full official-source raw comment views without reading label values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

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
from grain_research.vector_mechanisms.external_sources import exclusive_source_groups
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def peak_memory():
    """Report platform-explicit peak RSS when supported, otherwise unavailable."""
    try:
        import resource
    except ImportError:
        return {"value": None, "unit": None, "platform": sys.platform}
    return {
        "value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "unit": "bytes" if sys.platform == "darwin" else "KiB",
        "platform": sys.platform,
    }


def load_source_folds(root):
    """Read only IDs and each source's own fold; no psychological outcome column."""
    paths = {
        "big5": root / "pandora_official_big5_prepared.csv",
        "bridge": root / "bridge/pandora_official_bridge_strict377.csv",
    }
    paths.update(
        {
            a: root / "mbti_axes" / f"pandora_official_{a}_prepared.csv"
            for a in ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
        }
    )
    memberships, folds, meta = {}, {}, {}
    for name, path in paths.items():
        fold_col = "bridge_fold" if name == "bridge" else "official_fold"
        frame = pd.read_csv(path, usecols=["user_id", fold_col], dtype={"user_id": str})
        if (
            frame.user_id.isna().any()
            or frame.user_id.duplicated().any()
            or not frame[fold_col].isin(range(5)).all()
        ):
            raise ValueError("Invalid official ID/fold metadata")
        memberships[name] = set(frame.user_id)
        folds[name] = dict(zip(frame.user_id, frame[fold_col].astype(int), strict=True))
        meta[name] = {
            "path": str(path.relative_to(root)),
            "rows": len(frame),
            "columns_read": ["user_id", fold_col],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return memberships, folds, meta


def main():
    """Bound one raw scan, preserve all exclusions, and keep bridge out of head sources."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--previous-views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    memberships, folds, source_meta = load_source_folds(
        a.data_root / "prepared/pandora_official"
    )
    groups = exclusive_source_groups(memberships)
    all_authors = set().union(*groups.values())
    if len(all_authors) != sum(len(g) for g in groups.values()):
        raise ValueError("Source groups overlap")
    group_for = {u: name for name, users in groups.items() for u in users}
    previous = {
        b["author_id"] for b in json.loads(a.previous_views.read_text(encoding="utf-8"))
    }
    author_keys = {u: stable_key(u, ID_SEED)[:24] for u in all_authors}
    heaps, seen_ids = {u: [] for u in all_authors}, {u: set() for u in all_authors}
    counts = Counter()
    source = a.data_root / "PANDORA_official/all_comments_since_2015.csv"
    before = source.stat()
    manifest = {
        "protocol": "GRAIN_VM05_FULL_SOURCE_MATERIAL_V1",
        "protocol_commit": "0405492",
        "code_sha256": code_fingerprint(ROOT),
        "source_tables": source_meta,
        "source_counts": {k: len(v) for k, v in groups.items()},
        "candidates": len(all_authors),
        "source_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "units_per_view": 12,
        "views": 3,
        "reservoir": 192,
        "min_words": 40,
        "max_words": 100,
        "seed": COHORT_SEED,
        "max_seconds": 600,
        "psychological_values_read": False,
        "previous_cohort_id_count": len(previous),
        "previous_ids_excluded": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    start = time.monotonic()
    status = "RUNNING"
    write_json(out / "run_status.json", {"status": status})
    csv.field_size_limit(10_000_000)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {
            "author",
            "body",
            "lang",
            "created_utc",
            "id",
            "link_id",
            "subreddit",
        }.issubset(reader.fieldnames):
            raise ValueError("Required raw-comment fields missing")
        for r in reader:
            counts["rows_scanned"] += 1
            if counts["rows_scanned"] % 1_000_000 == 0:
                seconds = time.monotonic() - start
                print(
                    f"FULL_SOURCE rows={counts['rows_scanned']} seconds={seconds:.1f}",
                    flush=True,
                )
                write_json(
                    out / "run_status.json",
                    {
                        "status": "RUNNING",
                        "rows": counts["rows_scanned"],
                        "seconds": seconds,
                    },
                )
                if seconds > 600:
                    write_json(
                        out / "run_status.json",
                        {"status": "PARTIAL_BUDGET", "rows": counts["rows_scanned"]},
                    )
                    return
            u = r["author"]
            if u not in heaps:
                continue
            counts["source_comment_rows"] += 1
            if r["lang"] != "en":
                counts["non_english"] += 1
                continue
            words = len(r["body"].split())
            if not 40 <= words <= 100:
                counts["word_budget_excluded"] += 1
                continue
            if not r["id"] or not r["link_id"] or r["id"] in seen_ids[u]:
                counts["invalid_or_repeated_id"] += 1
                continue
            try:
                timestamp = float(r["created_utc"])
            except ValueError:
                timestamp = float("nan")
            if not 1_000_000_000 <= timestamp < 3_000_000_000:
                counts["invalid_timestamp"] += 1
                continue
            counts["eligible_comment_rows"] += 1
            rank = int(stable_key(r["id"], COHORT_SEED), 16)
            heap = heaps[u]
            if len(heap) == 192 and rank >= -heap[0][0]:
                continue
            comment = {
                "author_id": author_keys[u],
                "comment_id": stable_key(r["id"], ID_SEED)[:24],
                "thread_id": stable_key(r["link_id"], ID_SEED)[:24],
                "body": r["body"],
                "timestamp": timestamp,
                "word_count": words,
                "subreddit": r["subreddit"],
                "leakage_flags": leakage_flags(r["body"]),
            }
            entry = (-rank, r["id"], comment)
            if len(heap) == 192:
                old = heapq.heapreplace(heap, entry)
                seen_ids[u].remove(old[1])
            else:
                heapq.heappush(heap, entry)
            seen_ids[u].add(r["id"])
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw source changed during scan")
    first_owner, forbidden = {}, set()
    for heap in heaps.values():
        for _, _, c in heap:
            key = normalized_text_key(c["body"])
            owner = first_owner.setdefault(key, c["author_id"])
            if owner != c["author_id"]:
                forbidden.add(key)
    first_owner.clear()
    bundles, eligibility = [], []
    for u in sorted(all_authors, key=lambda u: stable_key(u, COHORT_SEED)):
        reason = "INSUFFICIENT_DISTINCT_COMMENTS"
        try:
            bundle = select_multicomment_bundle(
                [c for _, _, c in heaps[u]], units=12, forbidden_texts=forbidden
            )
        except ValueError as exc:
            if "temporal separation" not in str(exc):
                raise
            reason, bundle = "TIED_TEMPORAL_BOUNDARY", None
        row = {
            "author_id": author_keys[u],
            "source_group": group_for[u],
            "eligible": bundle is not None,
            "reason": "OK" if bundle else reason,
            "reservoir_comments": len(heaps[u]),
            "previous_mechanism_author": author_keys[u] in previous,
        }
        row.update({k + "_fold": f.get(u) for k, f in folds.items()})
        eligibility.append(row)
        if bundle:
            bundle.update(
                {
                    "source_group": group_for[u],
                    "split": "external_source_material",
                    "official_folds": {k: f[u] for k, f in folds.items() if u in f},
                    "previous_mechanism_author": row["previous_mechanism_author"],
                }
            )
            bundles.append(bundle)
    heaps.clear()
    seen_ids.clear()
    frame = pd.DataFrame(eligibility)
    frame.to_csv(out / "eligibility.csv", index=False)
    source_counts = []
    for name, users in groups.items():
        sub = frame[frame.source_group == name]
        source_counts.append(
            {
                "source_group": name,
                "source_authors": len(users),
                "eligible_authors": int(sub.eligible.sum()),
                "excluded_authors": int((~sub.eligible).sum()),
            }
        )
        write_json(
            out / f"{name}_author_views.json",
            [b for b in bundles if b["source_group"] == name],
        )
    pd.DataFrame(source_counts).to_csv(out / "source_counts.csv", index=False)
    fold_rows = []
    for name in folds:
        valid = frame[frame[name + "_fold"].notna()]
        for (group, fold), sub in valid.groupby(["source_group", name + "_fold"]):
            fold_rows.append(
                {
                    "source": name,
                    "source_group": group,
                    "fold": int(fold),
                    "source_authors": len(sub),
                    "eligible_authors": int(sub.eligible.sum()),
                }
            )
    pd.DataFrame(fold_rows).to_csv(out / "fold_counts.csv", index=False)
    if not bundles:
        raise ValueError("No eligible full-source materials")
    write_json(out / "author_views.json", bundles)
    inventory = {
        **dict(counts),
        **summarize_bundles(bundles),
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "cross_author_duplicate_text_keys": len(forbidden),
        "duplicate_scope": "retained reservoirs, exact normalized only",
        "previous_author_overlap": sum(b["previous_mechanism_author"] for b in bundles),
        "material_sha256": material_hash(bundles),
        "seconds": time.monotonic() - start,
        "peak_rss": peak_memory(),
        "psychological_values_read": False,
        "scope": "Full source scan, eligible multi-comment subset; not full official benchmark or a new lockbox",
    }
    write_json(out / "data_inventory.json", inventory)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "authors": len(bundles),
            "seconds": inventory["seconds"],
        },
    )
    print(
        json.dumps(
            {
                "source_counts": source_counts,
                "seconds": inventory["seconds"],
                "authors": len(bundles),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
