#!/usr/bin/env python3
"""Paired contrasts for `F06/state_attribution_r1` (seed-stratified author bootstrap)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/grain_gptpro_foundation_v3/F06/state_attribution_r1"


def main():
    """Write contrasts.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    units = defaultdict(dict)
    for r in csv.DictReader((args.run / "author_state_metrics.csv").open()):
        units[(r["seed"], r["ordinal"])][r["arm"]] = float(r["MSE"])
    keys = sorted(units)
    seeds = np.array([k[0] for k in keys])
    groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds.tolist()))]
    out = {}
    for a, b in (("scalar_state", "fixed"), ("state_z_only", "fixed"), ("fixed_all_h", "fixed"), ("scalar_state", "fixed_all_h"), ("scalar_state", "state_z_only")):
        d = np.array([units[k][a] - units[k][b] for k in keys])
        rng = np.random.default_rng(20260915)
        draws = np.array([d[np.concatenate([rng.choice(g, g.size) for g in groups])].mean() for _ in range(2000)])
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out[f"{a}-{b}"] = {"delta": float(d.mean()), "lo95": float(lo), "hi95": float(hi),
                           "decision": "SUPPORTED_LOWER_LOSS" if hi < 0 else ("HIGHER_LOSS" if lo > 0 else "INCONCLUSIVE")}
    (args.run / "contrasts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
