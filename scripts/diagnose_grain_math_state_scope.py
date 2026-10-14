#!/usr/bin/env python3
"""Preserved-data scope companion; candidates cannot become scored reports."""

import json

from run_grain_math_binding import read_bank
from run_grain_math_response import ROOT, sha, write_json

from grain_research.construction import math_state_scope as T


def main():
    """Replay frozen parser words without new calls or changing primary scores."""
    source = ROOT / "results/grain_math_response_v1/state_exposure_r1"
    dest = ROOT / "results/grain_math_response_v1/state_scope_diagnostic_r2"
    if (dest / "summary.json").exists():
        raise ValueError("scope diagnostic frozen")
    for name, digest in json.loads(
        (source / "04/artifact_hashes.json").read_text()
    ).items():
        if sha(source / name) != digest:
            raise ValueError("primary changed")
    rows = []
    for i, r in enumerate(read_bank(source / "00/documents.jsonl")):
        data = json.loads((source / f"01/parses/{i:04d}.json").read_text())
        rows.append(
            {"material_id": r["material_id"], "alternatives": T.alternatives(data)}
        )
    count = sum(not q["unique_scope"] for r in rows for q in r["alternatives"])
    invalid = sum(
        q["inside_candidate"]["status"] != "unresolved_scope"
        for r in rows
        for q in r["alternatives"]
        if not q["unique_scope"]
    )
    if invalid:
        raise ValueError("ambiguous candidate is eligible as a resolved observation")
    write_json(dest / "scope_alternatives.json", rows)
    write_json(
        dest / "summary.json",
        {
            "ambiguous_reports": count,
            "invalid_eligible_candidates": invalid,
            "new_calls": 0,
            "primary_scores_changed": False,
            "previous_scope_metadata": "confirmation_r1/01/primary_scope_alternatives.json retained",
            "current_code_sha256": sha(
                ROOT / "grain_research/construction/math_state_scope.py"
            ),
        },
    )
    write_json(
        dest / "artifact_hashes.json",
        {
            str(p.relative_to(dest)): sha(p)
            for p in dest.rglob("*")
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "ambiguous_reports": count,
                "invalid_eligible_candidates": invalid,
                "new_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
