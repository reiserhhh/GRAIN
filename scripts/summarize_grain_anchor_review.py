#!/usr/bin/env python3
"""Bind a provisional model issue review to exact private source matches."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import digest, write


def run(args):
    """Check complete reviewed identities and retain natural text in ignored files."""
    review = json.loads(args.review.read_text())
    if review["packet_sha256"] != digest(
        args.source / "private_blind_review_packet.json"
    ) or review["matches_sha256"] != digest(args.source / "private_matches.csv"):
        raise ValueError("Reviewed source changed")
    if review["rater_kind"] != "model" or review["independent"]:
        raise ValueError("This file is a non-independent model issue review")
    packet = pd.DataFrame(
        json.loads((args.source / "private_blind_review_packet.json").read_text())[
            "pairs"
        ]
    )
    ratings = pd.DataFrame(review["new_pairs"])
    if ratings.audit_id.duplicated().any() or set(ratings.audit_id) != set(
        packet.audit_id
    ):
        raise ValueError("Missing or duplicate reviewed cases")
    merged = packet.merge(ratings, on="audit_id", validate="one_to_one")
    merged.to_csv(args.review.parent / "private_reviewed_new_pairs.csv", index=False)
    matches = pd.read_csv(args.source / "private_matches.csv")
    old = matches[matches.bank == "ipip300"].set_index("candidate_index")
    new = matches[matches.bank == "ipip_plus_local360"].set_index("candidate_index")
    paired = old.join(new, lsuffix="_old", rsuffix="_new", how="inner")
    changed = paired[
        (paired.key_old != paired.key_new) | (paired.trait_old != paired.trait_new)
    ].reset_index()
    changed_review = pd.DataFrame(review["changed_pairs"])
    if changed_review.candidate_index.duplicated().any() or set(
        changed_review.candidate_index
    ) != set(changed.candidate_index):
        raise ValueError("Changed-key review does not cover the actual changes")
    changed.merge(changed_review, on="candidate_index", validate="one_to_one").to_csv(
        args.review.parent / "private_reviewed_changed_pairs.csv", index=False
    )
    summary = {
        "status": "PROVISIONAL_MODEL_REVIEW_COMPLETE",
        "new_pairs": len(merged),
        "changed_key_or_trait_pairs": len(changed),
        "issue_counts": ratings.issue.value_counts().to_dict(),
        "precision": None,
        "expert_validation": False,
        "independent": False,
        "feedback_promoted": 0,
        "review_sha256": digest(args.review),
        "scope": "Development error taxonomy; no criterion access, no correctness gold, no automatic scorer or anchor changes",
    }
    write(args.review.parent / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    run(parser.parse_args())
