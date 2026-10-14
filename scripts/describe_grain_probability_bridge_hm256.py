#!/usr/bin/env python3
"""Post-outcome ordinal description; primary writer/reader observations unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_probability_bridge_hm256 as H


def describe():
    """Retain all within-condition pairs, exact ties and both unselected contracts."""
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
    units = H.R.read(parent / "units.json")
    with np.load(parent / "analysis.npz") as z:
        a = {key: z[key] for key in z.files}
    groups = {}
    for row in units:
        groups.setdefault(row["condition"], []).append(row["unit_id"])
    condition_rows = []
    for condition, ids in groups.items():
        for k, contract in enumerate(("instruction", "disposition")):
            for reading in ("high_first", "low_first", "mean"):
                r = (
                    a["reverse_mean"][:, k]
                    if reading == "mean"
                    else a["reverse_orders"][
                        :, k, ("high_first", "low_first").index(reading)
                    ]
                )
                same, opposite, writer_tie, reader_tie, both_tie = 0, 0, 0, 0, 0
                for position, i in enumerate(ids):
                    for j in ids[position + 1 :]:
                        forward = float(a["source_odds"][i] - a["source_odds"][j])
                        reverse = float(r[i] - r[j])
                        if forward == 0 and reverse == 0:
                            both_tie += 1
                        elif forward == 0:
                            writer_tie += 1
                        elif reverse == 0:
                            reader_tie += 1
                        elif forward * reverse > 0:
                            same += 1
                        else:
                            opposite += 1
                condition_rows.append(
                    {
                        "condition": condition,
                        "axis": units[ids[0]]["axis"],
                        "contract": contract,
                        "reading": reading,
                        "same": same,
                        "opposite": opposite,
                        "writer_tie": writer_tie,
                        "reader_tie": reader_tie,
                        "both_tie": both_tie,
                        "pairs": len(ids) * (len(ids) - 1) // 2,
                    }
                )
    summary = []
    for axis in ("all", "extraversion", "neuroticism"):
        for contract in ("instruction", "disposition"):
            for reading in ("high_first", "low_first", "mean"):
                rows = [
                    r
                    for r in condition_rows
                    if (axis == "all" or r["axis"] == axis)
                    and r["contract"] == contract
                    and r["reading"] == reading
                ]
                counts = {
                    key: sum(r[key] for r in rows)
                    for key in (
                        "same",
                        "opposite",
                        "writer_tie",
                        "reader_tie",
                        "both_tie",
                        "pairs",
                    )
                }
                if counts["pairs"] != sum(
                    counts[key]
                    for key in (
                        "same",
                        "opposite",
                        "writer_tie",
                        "reader_tie",
                        "both_tie",
                    )
                ):
                    raise ValueError("Lost or duplicated ordinal pair")
                summary.append(
                    {
                        "axis": axis,
                        "contract": contract,
                        "reading": reading,
                        "conditions": len(rows),
                        **counts,
                        "same_fraction_all_pairs": counts["same"] / counts["pairs"],
                    }
                )
    out = parent / "ordinal_description_r1"
    out.mkdir(exist_ok=False)
    H.R.write(
        out / "manifest.json",
        {
            "post_outcome": True,
            "purpose": "ordering is distinct from probability compatibility; not accuracy or method selection",
            "code_sha256": H.R.digest(Path(__file__)),
            "parent_summary_sha256": H.R.digest(parent / "summary.json"),
            "parent_cycles_sha256": H.R.digest(parent / "cycles.csv"),
            "no_science_changes": True,
            "new_forwards": 0,
        },
    )
    H.R.table(out / "conditions.csv", condition_rows)
    H.R.write(
        out / "summary.json",
        {
            "ordinal_accounts": summary,
            "pairs_are_not_independent": True,
            "new_forwards": 0,
            "selection": False,
        },
    )
    H.R.seal(
        out, "output_seal.json", ("manifest.json", "conditions.csv", "summary.json")
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    describe()
