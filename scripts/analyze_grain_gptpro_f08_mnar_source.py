#!/usr/bin/env python3
"""Replay and paired contrasts for `F08/mnar_source_r1` (protocol v1)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"
REPLAY = {"oracle_source": "hard_channel_oracle_source", "gap_aware_read_source": "hard_channel_read_source"}
ARMS = ("gap_aware_read_source", "em_validation_law", "em_source_law")


def decide(lower, upper):
    """COMMON decision rule on a 95% interval for a loss difference."""
    if upper < 0:
        return "SUPPORTED_LOWER_LOSS"
    if lower > 0:
        return "HIGHER_LOSS"
    return "INCONCLUSIVE"


def main():
    """Write replay.json and paired_contrasts.csv into the run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=F08 / "mnar_source_r1")
    args = parser.parse_args()
    with (args.run / "metrics_by_author.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (F08 / "end_to_end_r1/metrics_by_author.csv").open() as f:
        archive = {
            (r["reader"], r["world"], r["seed"], r["author_id"], r["arm"], r["candidate"]): r
            for r in csv.DictReader(f)
            if r["panel"] == "mixed"
        }
    replay = {}
    for arm, old in REPLAY.items():
        diffs = [
            abs(float(r[k]) - float(archive[(r["reader"], r["world"], r["seed"], r["author_id"], old, r["candidate"])][k]))
            for r in rows
            if r["arm"] == arm
            for k in ("brier", "log_loss")
        ]
        replay[arm] = {"compared_values": len(diffs), "max_abs_diff": max(diffs)}
    units = defaultdict(dict)
    for r in rows:
        units[(r["reader"], r["world"], r["candidate"], r["seed"], r["author_id"])][r["arm"]] = float(r["brier"])
    out = []
    cells = sorted({k[:3] for k in units})
    for cell in cells:
        keys = sorted(k for k in units if k[:3] == cell)
        seeds = np.array([k[3] for k in keys])
        groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds))]
        base = np.array([units[k]["oracle_source"] for k in keys])
        for arm in ARMS:
            delta = np.array([units[k][arm] for k in keys]) - base
            rng = np.random.default_rng(20260915)
            draws = np.empty(2000)
            for b in range(2000):
                idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
                draws[b] = delta[idx].mean()
            lower, upper = np.quantile(draws, [0.025, 0.975])
            out.append(
                {
                    "reader": cell[0],
                    "world": cell[1],
                    "candidate": cell[2],
                    "arm": arm,
                    "oracle_source_brier": float(base.mean()),
                    "arm_brier": float(base.mean() + delta.mean()),
                    "delta_vs_oracle_source": float(delta.mean()),
                    "lower": float(lower),
                    "upper": float(upper),
                    "decision": decide(lower, upper),
                }
            )
    with (args.run / "paired_contrasts.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (args.run / "replay.json").write_text(json.dumps(replay, indent=1) + "\n")
    print(json.dumps(replay, indent=1))
    for r in out:
        print(r["reader"], r["world"], r["candidate"], r["arm"], f"{r['delta_vs_oracle_source']:+.5f} [{r['lower']:+.5f}, {r['upper']:+.5f}] {r['decision']}")


if __name__ == "__main__":
    main()
