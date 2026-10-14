#!/usr/bin/env python3
"""Replays and M contrasts for `F08/author_estep_r1` (protocol v1)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"


def boot(diff, seeds):
    """Seed-stratified author bootstrap 95% CI (2000 draws, seed 20260915)."""
    rng = np.random.default_rng(20260915)
    groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds.tolist()))]
    draws = np.array([diff[np.concatenate([rng.choice(g, g.size) for g in groups])].mean() for _ in range(2000)])
    return np.quantile(draws, [0.025, 0.975])


def decide(lo, hi):
    """COMMON rule."""
    return "SUPPORTED_LOWER_LOSS" if hi < 0 else ("HIGHER_LOSS" if lo > 0 else "INCONCLUSIVE")


def main():
    """Write replay.json and m_contrasts.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=F08 / "author_estep_r1")
    args = parser.parse_args()
    with (args.run / "metrics_by_author.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (F08 / "mnar_source_r1/metrics_by_author.csv").open() as f:
        prior = {(r["reader"], r["world"], r["seed"], r["author_id"], r["arm"], r["candidate"]): r for r in csv.DictReader(f)}
    units = defaultdict(dict)
    for r in rows:
        units[(r["reader"], r["world"], r["candidate"], r["seed"], r["author_id"])][r["arm"]] = r
    replay = {}
    for arm, old in (("oracle_source", "oracle_source"), ("em_pooled_validation_law", "em_validation_law")):
        d = [abs(float(r[k]) - float(prior[(r["reader"], r["world"], r["seed"], r["author_id"], old, r["candidate"])][k])) for r in rows if r["arm"] == arm for k in ("brier", "log_loss")]
        replay[f"{arm}_vs_mnar_source_r1"] = {"values": len(d), "max_abs_diff": max(d)}
    for law in ("validation_law", "source_law"):
        pooled_arm = "em_pooled_validation_law" if law == "validation_law" else None
        ref = {(r["reader"], r["world"], r["seed"], r["author_id"], r["candidate"]): r for r in prior.values() if r["arm"] == f"em_{law}"}
        d = [abs(float(v[f"em_author_{law}"][k]) - float((v[pooled_arm] if pooled_arm else ref[(key[0], key[1], key[3], key[4], key[2])])[k]))
             for key, v in units.items() if key[2] in ("b1", "b2") for k in ("brier", "log_loss")]
        replay[f"em_author_{law}_b1_b2_vs_pooled"] = {"values": len(d), "max_abs_diff": max(d)}
    out = []
    for reader in ("lexical", "nli"):
        for world in ("c2_xor", "c2_iid"):
            keys = sorted(k for k in units if k[:3] == (reader, world, "m"))
            seeds = np.array([k[3] for k in keys])
            pooled_src = np.array([float(prior[(k[0], k[1], k[3], k[4], "em_source_law", "m")]["brier"]) for k in keys])
            base = np.array([float(units[k]["oracle_source"]["brier"]) for k in keys])
            series = {
                "em_author_validation_law-oracle_source": np.array([float(units[k]["em_author_validation_law"]["brier"]) for k in keys]) - base,
                "em_author_source_law-oracle_source": np.array([float(units[k]["em_author_source_law"]["brier"]) for k in keys]) - base,
                "em_pooled_validation_law-oracle_source": np.array([float(units[k]["em_pooled_validation_law"]["brier"]) for k in keys]) - base,
                "em_author_validation_law-em_pooled_validation_law": np.array([float(units[k]["em_author_validation_law"]["brier"]) - float(units[k]["em_pooled_validation_law"]["brier"]) for k in keys]),
                "em_author_source_law-em_pooled_source_law": np.array([float(units[k]["em_author_source_law"]["brier"]) for k in keys]) - pooled_src,
            }
            for name, diff in series.items():
                lo, hi = boot(diff, seeds)
                out.append({"reader": reader, "world": world, "candidate": "m", "contrast": name, "delta": float(diff.mean()), "lower": float(lo), "upper": float(hi), "decision": decide(lo, hi)})
    with (args.run / "m_contrasts.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (args.run / "replay.json").write_text(json.dumps(replay, indent=1) + "\n")
    print(json.dumps(replay, indent=1))
    for r in out:
        print(r["reader"], r["world"], r["contrast"], f"{r['delta']:+.5f} [{r['lower']:+.5f},{r['upper']:+.5f}] {r['decision']}")


if __name__ == "__main__":
    main()
