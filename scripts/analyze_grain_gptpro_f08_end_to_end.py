#!/usr/bin/env python3
"""Replay check and paired author bootstrap for `F08/end_to_end_r1`.

Replay: arms with oracle-fitted sources must equal the archived downstream
application losses (lexical `downstream_full_r2`, NLI `nli_downstream_full_r1`).
Contrast: read-fitted minus oracle-fitted source, hard channel, Brier; paired
author bootstrap, 2000 draws, seed 20260915, stratified by seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"
ARCHIVE = {"lexical": F08 / "downstream_full_r2", "nli": F08 / "nli_downstream_full_r1"}
ARM_CHANNEL = {"oracle_channel_oracle_source": "oracle", "hard_channel_oracle_source": "hard"}


def decide(lower, upper):
    """COMMON decision rule on a 95% interval for a loss difference."""
    if upper < 0:
        return "SUPPORTED_LOWER_LOSS"
    if lower > 0:
        return "HIGHER_LOSS"
    return "INCONCLUSIVE"


def main():
    """Write replay.json and paired_contrasts_recomputed.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=F08 / "end_to_end_r1")
    args = parser.parse_args()
    with (args.run / "metrics_by_author.csv").open() as f:
        rows = list(csv.DictReader(f))
    replay = {}
    for reader, folder in ARCHIVE.items():
        with (folder / "metrics_by_author.csv").open() as f:
            archive = {
                (r["world"], r["seed"], r["panel"], r["author_id"], r["channel"], r["candidate"]): r
                for r in csv.DictReader(f)
                if r["role"] == "application"
            }
        for arm, channel in ARM_CHANNEL.items():
            diffs, missing = [], 0
            for r in rows:
                if r["reader"] != reader or r["arm"] != arm:
                    continue
                key = (r["world"], r["seed"], r["panel"], r["author_id"], channel, r["candidate"])
                if key not in archive:
                    missing += 1
                    continue
                a = archive[key]
                diffs.append(abs(float(r["brier"]) - float(a["brier"])))
                diffs.append(abs(float(r["log_loss"]) - float(a["log_loss"])))
            replay[f"{reader}:{arm}"] = {
                "compared_rows": len(diffs) // 2,
                "unmatched_rows": missing,
                "max_abs_diff": max(diffs) if diffs else None,
            }
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["arm"] == "oracle_channel_oracle_source":
            continue
        cell = (r["reader"], r["world"], r["panel"], r["candidate"])
        cells[cell][(r["seed"], r["author_id"])][r["arm"]] = float(r["brier"])
    out = []
    for cell in sorted(cells):
        units = cells[cell]
        keys = sorted(units)
        oracle = np.array([units[k]["hard_channel_oracle_source"] for k in keys])
        read = np.array([units[k]["hard_channel_read_source"] for k in keys])
        delta = read - oracle
        seeds = np.array([k[0] for k in keys])
        rng = np.random.default_rng(20260915)
        groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds))]
        draws = np.empty(2000)
        for b in range(2000):
            idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
            draws[b] = delta[idx].mean()
        lower, upper = np.quantile(draws, [0.025, 0.975])
        out.append(
            {
                "reader": cell[0],
                "world": cell[1],
                "panel": cell[2],
                "candidate": cell[3],
                "oracle_source_brier": float(oracle.mean()),
                "read_source_brier": float(read.mean()),
                "delta": float(delta.mean()),
                "lower": float(lower),
                "upper": float(upper),
                "decision": decide(lower, upper) if np.any(delta != 0) else "INCONCLUSIVE",
            }
        )
    with (args.run / "paired_contrasts_recomputed.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (args.run / "replay.json").write_text(json.dumps(replay, indent=1) + "\n")
    print(json.dumps(replay, indent=1))


if __name__ == "__main__":
    main()
