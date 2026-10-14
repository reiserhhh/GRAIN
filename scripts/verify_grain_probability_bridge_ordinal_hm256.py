#!/usr/bin/env python3
"""Replay the post-outcome pair description from separately sealed cycle rows."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_probability_bridge_hm256 as H


def verify():
    """Compare every ordinal count with primary signed rows, not the same loop."""
    parent = H.DEFAULT
    H.R.check(
        parent,
        (
            "cold_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verified_seal.json",
        ),
    )
    out = parent / "ordinal_description_r1"
    manifest = H.R.read(out / "manifest.json")
    if (
        H.R.digest(ROOT / "scripts/describe_grain_probability_bridge_hm256.py")
        != manifest["code_sha256"]
    ):
        raise ValueError("Changed description code")
    for path, checksum in H.R.read(out / "output_seal.json").items():
        if H.R.digest(out / path) != checksum:
            raise ValueError("Changed ordinal description")
    counts = {}
    with (parent / "cycles.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["unit_i"]) >= int(row["unit_j"]):
                continue
            key = row["condition"], row["contract"], row["reading"]
            counter = counts.setdefault(key, Counter())
            forward, reverse = (
                float(row["writer_difference"]),
                float(row["reverse_difference"]),
            )
            category = (
                "both_tie"
                if forward == reverse == 0
                else "writer_tie"
                if forward == 0
                else "reader_tie"
                if reverse == 0
                else "same"
                if (forward > 0) == (reverse > 0)
                else "opposite"
            )
            counter[category] += 1
            counter["pairs"] += 1
    error, accounts = 0, 0
    with (out / "conditions.csv").open() as f:
        for row in csv.DictReader(f):
            key = row["condition"], row["contract"], row["reading"]
            for name in (
                "same",
                "opposite",
                "writer_tie",
                "reader_tie",
                "both_tie",
                "pairs",
            ):
                error = max(error, abs(int(row[name]) - counts[key][name]))
                accounts += 1
    result = {
        "counter_error": error,
        "condition_accounts": accounts,
        "independent_cycle_row_route": True,
        "same_agent_not_blind": True,
        "new_forwards": 0,
        "verifier_sha256": H.R.digest(Path(__file__)),
    }
    H.R.write(out / "verification.json", result)
    if error:
        raise ValueError("Ordinal report disagrees with primary cycle rows")
    H.R.seal(out, "verified_seal.json", ("verification.json",))
    print(json.dumps(result))


if __name__ == "__main__":
    verify()
