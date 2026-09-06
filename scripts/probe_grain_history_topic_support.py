#!/usr/bin/env python3
"""Outcome-blind shared-subreddit donor feasibility; never replace primary histories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.history_instrument import (
    archive_instrument,
    tokenizer_matches,
)
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.audit_grain_source_history_material import (
    independent_match,
    independent_view,
)
from scripts.prepare_grain_source_history import reference_ids
from scripts.run_grain_vector_mechanism_controls import write_json


def overlap_match(profile, current, whole, candidates):
    """Require shared metadata, then preserve original time/provenance/cost rules."""
    eligible = [
        d for d in candidates if profile["subreddits"] & d["profile"]["subreddits"]
    ]
    if not eligible:
        return None
    try:
        return independent_match(profile, current, whole, eligible)
    except ValueError as e:
        if str(e) != "Supported case has no eligible prior reference":
            raise
        return None


def main():
    """Audit all strict-supported views without psychological outcomes or forwards."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "material-dir",
        "audit-dir",
        "views",
        "preflight-dir",
        "qwen-tokenizer",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=900)
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 1800:
        raise ValueError("Require bounded runtime")
    start = time.monotonic()
    m, _ = verified_packet(a.material_dir)
    am, _ = verified_packet(a.audit_dir)
    source = json.loads(a.views.read_text())
    if (
        material_hash(source) != m["common_material_sha256"]
        or am["material_identity"] != m["identity"]
        or m["labels_read"]
        or am["labels_read"]
        or digest(a.preflight_dir / "budget_by_author.csv") != m["budget_sha256"]
    ):
        raise ValueError("Changed material/audit/source boundary")
    tokenizer_matches(a.qwen_tokenizer, m)
    tok = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    refs = reference_ids(source, budget)
    candidates = []
    for b in source:
        if b["author_id"] not in refs:
            continue
        for vi, v in enumerate(b["views"]):
            h, _, pf = independent_view(v, tok)
            candidates.append(
                {"author_id": b["author_id"], "view": vi, "history": h, "profile": pf}
            )
    if len(candidates) != m["dictionary_views"]:
        raise ValueError("Dictionary changed")
    table = pd.read_csv(
        a.audit_dir / "private_strict_history_registry.csv", dtype={"author_id": str}
    )
    allowed = set(table.loc[table.strict_no_near_copy, "author_id"])
    cases = json.loads((a.material_dir / "private_history_cases.json").read_text())
    by_id = {b["author_id"]: b for b in source}
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private result directory required")
    out.mkdir(parents=True, exist_ok=False)
    archive_instrument(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_SUBREDDIT_OVERLAP_FEASIBILITY_V1",
        "material_identity": m["identity"],
        "audit_identity": am["identity"],
        "rule": "At least one shared subreddit; original minimum eligible cost",
        "script_sha256": digest(Path(__file__)),
        "audit_match_code_sha256": digest(
            ROOT / "scripts/audit_grain_source_history_material.py"
        ),
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "max_seconds": a.max_seconds,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "neural_forwards": 0,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    rows = []
    for case in cases:
        if case["author_id"] not in allowed:
            continue
        b = by_id[case["author_id"]]
        whole = [c for v in b["views"] for c in v]
        for saved in case["views"]:
            if time.monotonic() - start > a.max_seconds:
                pd.DataFrame(rows).to_csv(
                    out / "private_partial_views.csv", index=False
                )
                write_json(
                    out / "run_status.json",
                    {"status": "PARTIAL_TIME_BOUND", "views": len(rows)},
                )
                raise TimeoutError("Keep partial feasibility, never change primary")
            _, current, pf = independent_view(b["views"][saved["view"]], tok)
            choice = overlap_match(pf, current, whole, candidates)
            r = {
                "author_id": b["author_id"],
                "source_group": b["source_group"],
                "view": saved["view"],
                "overlap_supported": choice is not None,
                "primary_jaccard": saved["match"]["subreddit_jaccard"],
            }
            if choice is not None:
                (cost, uid, vi, _, length, gap, j), n = choice
                r.update(
                    donor_author_id=uid,
                    donor_view=vi,
                    eligible_donors=n,
                    subreddit_jaccard=j,
                    length_error=length,
                    time_error=gap,
                    cost_change=cost - saved["match"]["cost"],
                    length_error_change=length - saved["match"]["length_error"],
                    time_error_change=gap - saved["match"]["time_error"],
                )
                if j <= 0 or r["cost_change"] < -1e-12:
                    raise ValueError(
                        "Constrained donor cannot beat unrestricted minimum"
                    )
            rows.append(r)
    f = pd.DataFrame(rows)
    if len(f) != len(allowed) * 3:
        raise ValueError("Incomplete feasibility coverage")
    f.to_csv(out / "private_candidate_views.csv", index=False)
    report = []
    for group, g in f.groupby("source_group"):
        supported = g.loc[g.overlap_supported]
        full = g.groupby("author_id").overlap_supported.all()
        r = {
            "source_group": group,
            "authors": len(full),
            "views": len(g),
            "overlap_supported_views": len(supported),
            "complete_authors": int(full.sum()),
        }
        for field in (
            "subreddit_jaccard",
            "length_error",
            "time_error",
            "cost_change",
            "length_error_change",
            "time_error_change",
            "eligible_donors",
        ):
            for q, name in ((0.5, "median"), (0.9, "p90")):
                r[field + "_" + name] = (
                    float(np.quantile(supported[field], q)) if len(supported) else None
                )
        report.append(r)
    pd.DataFrame(report).to_csv(out / "overlap_support.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "groups": report,
        "labels_read": False,
        "neural_forwards": 0,
        "primary_material_modified": False,
        "new_donor_layouts_qualified": False,
        "new_donor_near_copy_audit_complete": False,
        "seconds": time.monotonic() - start,
        "outputs_sha256": {
            n: digest(out / n)
            for n in ("private_candidate_views.csv", "overlap_support.csv")
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
