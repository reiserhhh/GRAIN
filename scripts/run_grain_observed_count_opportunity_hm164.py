#!/usr/bin/env python3
"""Account for actual expression occurrence versus denominator opportunities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_count_opportunity as M
from grain_research.construction.math_observed_text_anchor import persona_pairs
from scripts.run_grain_observed_text_anchor_hm164 import digest, read, table, write

PARENT = ROOT / "results/grain_math_response_v1/observed_text_anchor_r1"
OUT = PARENT / "01_count_opportunity_account_r1"
SCIENCE = (
    "grain_research/construction/math_observed_count_opportunity.py",
    "scripts/run_grain_observed_count_opportunity_hm164.py",
    "tests/test_grain_observed_count_opportunity.py",
    "docs/GRAIN_OBSERVED_COUNT_OPPORTUNITY_FOLLOWUP_HM164.md",
    "grain_research/construction/math_observed_text_anchor.py",
    "scripts/run_grain_observed_text_anchor_hm164.py",
)


def prepare():
    """Pin the already published primary result and a distinct follow-up design."""
    config = read(PARENT / "manifest.json")["config"]
    files = [ROOT / p for p in SCIENCE] + [
        PARENT / "summary.json",
        PARENT / "observed_counts.csv",
        ROOT / config["source"] / "materials.json",
    ]
    OUT.mkdir(exist_ok=False)
    write(
        OUT / "manifest.json",
        {
            "status": "POST_OUTCOME_SEALED_BEFORE_FOLLOWUP",
            "config": config,
            "hashes": {str(p.relative_to(ROOT)): digest(p) for p in files},
        },
    )
    print(json.dumps({"prepared": str(OUT)}))


def analyze():
    """Use every actual count pair and verify the identity against rate subtraction."""
    start = time.monotonic()
    manifest = read(OUT / "manifest.json")
    for name, expected in manifest["hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Follow-up dependency changed")
    config = manifest["config"]
    rows = read(ROOT / config["source"] / "materials.json")
    definitions = config["observations"]
    counts, words = {}, {}
    with (PARENT / "observed_counts.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["text_id"]
            counts.setdefault(key, {})[row["anchor"]] = int(row["count"])
            word = max(int(row["words"]), 1)
            if key in words and words[key] != word:
                raise ValueError("Inconsistent word opportunities")
            words[key] = word
    pairs, meta = persona_pairs(rows)
    ch = np.array(
        [
            [counts[rows[i]["text_id"]][d["id"]] for d in definitions]
            for i in pairs[:, 0]
        ]
    )
    cl = np.array(
        [
            [counts[rows[i]["text_id"]][d["id"]] for d in definitions]
            for i in pairs[:, 1]
        ]
    )
    wh = np.array([words[rows[i]["text_id"]] for i in pairs[:, 0]])[:, None]
    wl = np.array([words[rows[i]["text_id"]] for i in pairs[:, 1]])[:, None]
    account = M.count_opportunity_account(ch, cl, wh, wl)
    effects = []
    for i, m in enumerate(meta):
        for j, definition in enumerate(definitions):
            effects.append(
                m
                | {
                    "anchor": definition["id"],
                    "high_words": int(wh[i, 0]),
                    "low_words": int(wl[i, 0]),
                    **{key: float(value[i, j]) for key, value in account.items()},
                }
            )
    table(OUT / "paired_accounts.csv", effects)
    conditions = []
    for axis, task, style in sorted(
        {(m["axis"], m["situation"], m["style"]) for m in meta}
    ):
        mask = np.array(
            [
                (m["axis"], m["situation"], m["style"]) == (axis, task, style)
                for m in meta
            ]
        )
        for j, definition in enumerate(definitions):
            conditions.append(
                {
                    "axis": axis,
                    "situation": task,
                    "style": style,
                    "anchor": definition["id"],
                    "pairs": int(mask.sum()),
                    **{
                        f"mean_{key}": float(value[mask, j].mean())
                        for key, value in account.items()
                    },
                }
            )
    table(OUT / "condition_accounts.csv", conditions)
    error = float(np.max(abs(account["reconstruction_error"])))
    if error > 1e-12:
        raise ValueError("Signed rate account failed")
    result = {
        "status": "COMPLETE",
        "post_outcome_followup": True,
        "pairs": len(pairs),
        "accounts": len(effects),
        "reconstruction_error": error,
        "count_vs_rate_opposite": int(
            (
                account["raw_count_difference"] * account["rate_difference"] < -1e-12
            ).sum()
        ),
        "rate_nonzero_count_zero": int(
            (
                (account["raw_count_difference"] == 0)
                & (abs(account["rate_difference"]) > 1e-12)
            ).sum()
        ),
        "elapsed_seconds": time.monotonic() - start,
        "new_neural_forwards": 0,
        "original_scope_complete": False,
    }
    write(OUT / "summary.json", result)
    write(
        OUT / "output_seal.json",
        {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()},
    )
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.stage]()
