#!/usr/bin/env python3
"""Summaries and paired contrasts for `F13/recovery3_r1` (protocol v1)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/grain_gptpro_foundation_v3/F13/recovery3_r1"
FIELDS = ("se", "bias", "brier", "cover", "width", "state", "personal", "source", "event", "missing_rate")


def boot(values, seeds, draws=2000):
    """Seed-stratified author bootstrap 95% CI of a mean."""
    rng = np.random.default_rng(20260915)
    groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds.tolist()))]
    stats = np.array([values[np.concatenate([rng.choice(g, g.size) for g in groups])].mean() for _ in range(draws)])
    return np.quantile(stats, [0.025, 0.975])


def decide(lo, hi):
    """COMMON rule."""
    return "SUPPORTED_LOWER_LOSS" if hi < 0 else ("HIGHER_LOSS" if lo > 0 else "INCONCLUSIVE")


def main():
    """Write summary.csv, contrasts.csv, summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    with (args.run / "author_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    cell = defaultdict(list)
    for r in rows:
        cell[(r["world"], r["mechanism"], r["candidate"])].append(r)
    summary = []
    for (world, mech, cand), rs in sorted(cell.items()):
        seeds = np.array([r["seed"] for r in rs])
        out = {"world": world, "mechanism": mech, "candidate": cand, "authors": len(rs)}
        for k in FIELDS:
            v = np.array([float(r[k]) for r in rs])
            out[k] = float(v.mean())
            if k in ("cover", "se", "bias", "width"):
                lo, hi = boot(v, seeds)
                out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
        summary.append(out)
    units = defaultdict(dict)
    for r in rows:
        units[(r["world"], r["candidate"], r["seed"], r["author_id"])][r["mechanism"]] = r
    contrasts = []
    for world in ("c2_xor", "c2_iid"):
        for cand in ("b1", "b2", "m"):
            keys = sorted(k for k in units if k[:2] == (world, cand))
            seeds = np.array([k[2] for k in keys])
            for mech in ("MCAR", "MAR", "MNAR"):
                d = np.array([float(units[k][mech]["se"]) - float(units[k]["none"]["se"]) for k in keys])
                lo, hi = boot(d, seeds)
                contrasts.append({"world": world, "candidate": cand, "contrast": f"{mech}-none:se_vs_true_next_p", "delta": float(d.mean()), "lower": float(lo), "upper": float(hi), "decision": decide(lo, hi)})
            for mech in ("MAR", "MNAR"):
                d = np.array([float(units[k][mech]["bias"]) - float(units[k]["MCAR"]["bias"]) for k in keys])
                lo, hi = boot(d, seeds)
                contrasts.append({"world": world, "candidate": cand, "contrast": f"{mech}-MCAR:bias", "delta": float(d.mean()), "lower": float(lo), "upper": float(hi), "decision": "DIFFERENT" if lo > 0 or hi < 0 else "INCONCLUSIVE"})
    for name, data in (("summary.csv", summary), ("contrasts.csv", contrasts)):
        with (args.run / name).open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for s in summary:
        print(s["world"], s["mechanism"], s["candidate"], f"miss {s['missing_rate']:.3f} se {s['se']:.5f} bias {s['bias']:+.4f} cover {s['cover']:.3f} [{s['cover_lo']:.3f},{s['cover_hi']:.3f}] width {s['width']:.3f} state {s['state']:.4f} pers {s['personal']:.4f} src {s['source']:.5f}")
    for c in contrasts:
        print(c["world"], c["candidate"], c["contrast"], f"{c['delta']:+.5f} [{c['lower']:+.5f},{c['upper']:+.5f}] {c['decision']}")


if __name__ == "__main__":
    main()
