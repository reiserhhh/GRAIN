#!/usr/bin/env python3
"""Freeze an analyst's source-only semantic decisions before model readouts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.memory import (
    recover_summary,
    validate_summary_review,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Reject missing ratings and save exact unedited facts with provenance."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources, generated, decisions = map(read, (a.sources, a.generated, a.decisions))
    if a.output.exists():
        raise ValueError("Never overwrite a frozen pre-readout review")
    by_id = {r["author_id"]: r["history"] for r in sources}
    if len(decisions["authors"]) != len(generated):
        raise ValueError("Decision author count mismatch")
    reviews = []
    for i, row in enumerate(generated):
        audit = recover_summary(row["raw"], by_id[row["author_id"]])
        ratings = decisions["authors"][i]
        if len(ratings) != len(audit["facts"]):
            raise ValueError("Decision count mismatch")
        for fact, rating in zip(audit["facts"], ratings, strict=True):
            fact["semantic_review"] = rating["decision"]
            fact["review_reason"] = rating["reason"]
        reviews.append({"author_id": row["author_id"], "audit": audit})
    validate_summary_review(generated, reviews, sources)
    output = {
        "completed_before_readout": True,
        "generated_sha256": material_hash(generated),
        "source_sha256": material_hash(sources),
        "decisions_sha256": material_hash(decisions),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "One Codex research analyst; no independent inter-rater claim",
        "review_basis": "Generated statement plus complete cited source; no downstream logits or psychological labels",
        "protocol": decisions["protocol"],
        "reviews": reviews,
    }
    write_json(a.output, output)
    print(
        json.dumps(
            {
                "authors": len(reviews),
                "facts": sum(len(r["audit"]["facts"]) for r in reviews),
                "accepted": sum(
                    f["semantic_review"] == "ACCEPT"
                    for r in reviews
                    for f in r["audit"]["facts"]
                ),
                "review_sha256": material_hash(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
