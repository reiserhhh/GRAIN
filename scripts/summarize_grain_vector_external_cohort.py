#!/usr/bin/env python3
"""Independently verify VM05 material identities, membership, folds and exclusions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import (
    ID_SEED,
    validate_multicomment_bundle,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import normalized_text_key, stable_key
from grain_research.vector_mechanisms.external_sources import exclusive_source_groups
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_cohort import load_source_folds


def validate_materials(bundles, eligibility, memberships, folds):
    """Validate complete source-ledger coverage independently of extraction counters."""
    groups = exclusive_source_groups(memberships)
    ids = {stable_key(u, ID_SEED)[:24]: u for u in set().union(*groups.values())}
    if len(ids) != sum(map(len, groups.values())):
        raise ValueError("Source overlap or pseudonym collision")
    if eligibility.author_id.duplicated().any() or set(eligibility.author_id) != set(
        ids
    ):
        raise ValueError("Incomplete or duplicated source eligibility ledger")
    expected = {
        k: group
        for group, users in groups.items()
        for k, u in ids.items()
        if u in users
    }
    for row in eligibility.to_dict("records"):
        if row["source_group"] != expected[row["author_id"]]:
            raise ValueError("Source group mismatch")
        for name, mapping in folds.items():
            actual = row[name + "_fold"]
            value = mapping.get(ids[row["author_id"]])
            if (value is None and not pd.isna(actual)) or (
                value is not None and value != actual
            ):
                raise ValueError("Source-specific fold mismatch")
    seen, comments, texts, threads = set(), set(), {}, {}
    rows = []
    for b in bundles:
        uid = b["author_id"]
        if uid in seen or uid not in ids or b["source_group"] != expected[uid]:
            raise ValueError("Invalid author material membership")
        seen.add(uid)
        validate_multicomment_bundle(b, units=12)
        if b["official_folds"] != {
            k: m[ids[uid]] for k, m in folds.items() if ids[uid] in m
        }:
            raise ValueError("Bundle source folds differ")
        records = [c for v in b["views"] for c in v]
        flagged = 0
        for c in records:
            if c["comment_id"] in comments:
                raise ValueError("Comment shared across authors")
            comments.add(c["comment_id"])
            key = normalized_text_key(c["body"])
            if texts.setdefault(key, uid) != uid:
                raise ValueError("Normalized text shared across authors")
            if (
                len(c["body"].split()) != c["word_count"]
                or not 40 <= c["word_count"] <= 100
            ):
                raise ValueError("Word budget mismatch")
            flagged += any(c["leakage_flags"].values())
            threads.setdefault(c["thread_id"], set()).add(b["source_group"])
        rows.append(
            {
                "source_group": b["source_group"],
                "flagged_comments": flagged,
                "comments": len(records),
                "has_flags": flagged > 0,
                "words": sum(c["word_count"] for c in records),
                "span_days": (records[-1]["timestamp"] - records[0]["timestamp"])
                / 86400,
            }
        )
    if seen != set(eligibility.loc[eligibility.eligible, "author_id"]):
        raise ValueError("Eligible authors do not match material authors")
    return pd.DataFrame(rows), {
        "source_authors": len(ids),
        "eligible_authors": len(seen),
        "comments": len(comments),
        "source_author_overlap": 0,
        "cross_author_comment_overlap": 0,
        "cross_author_normalized_text_overlap": 0,
        "cross_source_shared_discussion_threads": sum(
            len(v) > 1 for v in threads.values()
        ),
        "bridge_source_shared_discussion_threads": sum(
            "bridge" in v and len(v) > 1 for v in threads.values()
        ),
        "source_fold_validation": "PASS",
        "temporal_and_within_author_thread_checks": "PASS",
    }


def main():
    """Emit aggregate quality findings without raw IDs, source quotes or trait values."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda name: json.loads((a.parent_dir / name).read_text(encoding="utf-8"))
    manifest, status, inventory = (
        read("manifest.json"),
        read("run_status.json"),
        read("data_inventory.json"),
    )
    if status["status"] != "COMPLETE":
        raise ValueError("Incomplete source scan")
    bundles = read("author_views.json")
    if material_hash(bundles) != inventory["material_sha256"]:
        raise ValueError("Material hash changed")
    members, folds, metadata = load_source_folds(
        a.data_root / "prepared/pandora_official"
    )
    if metadata != manifest["source_tables"]:
        raise ValueError("Official source files changed")
    eligibility = pd.read_csv(a.parent_dir / "eligibility.csv")
    frame, checks = validate_materials(bundles, eligibility, members, folds)
    rows = []
    for group, sub in frame.groupby("source_group"):
        rows.append(
            {
                "source_group": group,
                "authors": len(sub),
                "comments": int(sub.comments.sum()),
                "any_flagged_comments": int(sub.flagged_comments.sum()),
                "authors_with_flags": int(sub.has_flags.sum()),
                "median_words_per_author": float(sub.words.median()),
                "span_days_p10_p50_p90": np.quantile(
                    sub.span_days, [0.1, 0.5, 0.9]
                ).tolist(),
            }
        )
    mbti = set().union(*(v for k, v in members.items() if k.endswith("_cont")))
    bridge_extra = members["bridge"] - mbti
    checks["bridge_extra_outside_mbti"] = len(bridge_extra)
    checks["bridge_extra_in_big5"] = len(bridge_extra & members["big5"])
    checks["bridge_extra_outside_both_sources"] = len(bridge_extra - members["big5"])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use new audit directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    pd.DataFrame(rows).to_csv(a.output_dir / "coverage_and_flags.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "material_sha256": inventory["material_sha256"],
        "checks": checks,
        "groups": rows,
        "exclusion_counts": dict(
            Counter(eligibility.loc[~eligibility.eligible, "reason"])
        ),
        "old_cohort_eligible_overlap": int(
            eligibility.loc[eligibility.eligible, "previous_mechanism_author"].sum()
        ),
        "scope": "Material verification only; matched conditions and psychological outcomes not evaluated",
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
