#!/usr/bin/env python3
"""Summarize audited history support, repair invariants and imperfect donor matching."""

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

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.history_instrument import archive_instrument
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Use metadata only; no new embeddings, labels, donors or matching thresholds."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-material", "material-dir", "audit-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    oldm, olds = verified_packet(a.previous_material)
    m, s = verified_packet(a.material_dir)
    am, audit = verified_packet(a.audit_dir)
    if (
        oldm["common_material_sha256"] != m["common_material_sha256"]
        or oldm["reference_commitment"] != m["reference_commitment"]
        or oldm["matching_rule"] != m["matching_rule"]
        or m.get("history_envelope") != "body_and_separator"
        or am["material_identity"] != m["identity"]
        or audit["material_identity"] != m["identity"]
        or any(x["labels_read"] for x in (oldm, m, am, audit))
    ):
        raise ValueError(
            "History versions/roles/audit differ beyond the declared repair"
        )
    old = json.loads((a.previous_material / "private_history_cases.json").read_text())
    new = json.loads((a.material_dir / "private_history_cases.json").read_text())
    prior = {r["author_id"]: r for r in old}
    current = {r["author_id"]: r for r in new}
    if not prior.keys() <= current.keys():
        raise ValueError("Previously supported author lost")
    checked = 0
    for uid, record in prior.items():
        for before, after in zip(record["views"], current[uid]["views"], strict=True):
            if {k: v for k, v in before.items() if k != "layouts"} != {
                k: v for k, v in after.items() if k != "layouts"
            }:
                raise ValueError("Boundary repair changed history material or donor")
            checked += 1
    strict = pd.read_csv(
        a.audit_dir / "private_strict_history_registry.csv", dtype={"author_id": str}
    )
    support = pd.read_csv(
        a.material_dir / "private_support_by_author.csv", dtype={"author_id": str}
    )
    allowed = set(strict.loc[strict.strict_no_near_copy, "author_id"])
    if not allowed <= current.keys():
        raise ValueError("Strict registry escaped original support")
    cohorts, rows, reuse = [], [], Counter()
    for group, f in support.groupby("source_group"):
        cohorts.append(
            {
                "source_group": group,
                "target_authors": len(f),
                "old_supported": olds["supported_by_group"][group],
                "repaired_supported": s["supported_by_group"][group],
                "strict_supported": len(set(f.author_id) & allowed),
                "strict_same_ready": int(
                    (f.author_id.isin(allowed) & f.original_all_arms_ready).sum()
                ),
            }
        )
    for case in new:
        if case["author_id"] not in allowed:
            continue
        for view in case["views"]:
            reuse[view["donor_author_id"]] += 1
            raw = [
                r
                for r in view["layouts"]
                if r["text_arm"] == "raw"
                and r["task"] == "behavior"
                and r["history_arm"] == "own"
            ]
            if len(raw) != 1:
                raise ValueError("Ambiguous primary history layout")
            rows.append(
                {
                    "source_group": case["source_group"],
                    **view["match"],
                    **{
                        k: raw[0][k]
                        for k in (
                            "prompt_tokens",
                            "current_tokens",
                            "history_slot_tokens",
                        )
                    },
                }
            )
    quality = []
    frame = pd.DataFrame(rows)
    for group, f in frame.groupby("source_group"):
        for field in (
            "length_error",
            "time_error",
            "subreddit_jaccard",
            "eligible_donors",
            "prompt_tokens",
            "current_tokens",
            "history_slot_tokens",
        ):
            q = np.quantile(f[field], [0.1, 0.5, 0.9])
            quality.append(
                {
                    "source_group": group,
                    "field": field,
                    "views": len(f),
                    "p10": float(q[0]),
                    "median": float(q[1]),
                    "p90": float(q[2]),
                }
            )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Require private result directory")
    out.mkdir(parents=True, exist_ok=False)
    archive_instrument(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_HISTORY_MATERIAL_COMPARISON_V1",
        "previous_identity": oldm["identity"],
        "material_identity": m["identity"],
        "audit_identity": am["identity"],
        "script_sha256": digest(Path(__file__)),
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    pd.DataFrame(cohorts).to_csv(out / "support_comparison.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "matching_quality.csv", index=False)
    shares = np.asarray(list(reuse.values()), dtype=float) / sum(reuse.values())
    result = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "same_material_views_verified": checked,
        "recovered_authors": len(current.keys() - prior.keys()),
        "lost_authors": len(prior.keys() - current.keys()),
        "strict_supported_authors": len(allowed),
        "strict_views": len(frame),
        "dictionary_authors_used": len(reuse),
        "max_dictionary_author_share": float(shares.max()),
        "effective_dictionary_authors_inverse_simpson": float(1 / np.sum(shares**2)),
        "labels_read": False,
        "model_forwards": 0,
        "scope": "Material support and conditional fixed-dictionary match quality, not prediction or psychometric validity",
        "outputs_sha256": {
            name: digest(out / name)
            for name in ("support_comparison.csv", "matching_quality.csv")
        },
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
