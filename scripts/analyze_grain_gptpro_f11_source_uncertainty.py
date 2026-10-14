#!/usr/bin/env python3
"""Coverage summaries with author-cluster CIs for `F11/source_uncertainty_r1`."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/grain_gptpro_foundation_v3/F11/source_uncertainty_r1"
ARMS = ("plugin", "augmented", "mixture", "oracle")


def boot(values, seeds):
    """Seed-stratified author-cluster bootstrap 95% CI of a mean."""
    rng = np.random.default_rng(20260915)
    groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds.tolist()))]
    stats = np.array([values[np.concatenate([rng.choice(g, g.size) for g in groups])].mean() for _ in range(2000)])
    return np.quantile(stats, [0.025, 0.975])


def wilson(k, n, z=1.959963984540054):
    """Wilson 95% for pooled intervals (treats traits as independent; secondary)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    """Write coverage_summary.csv and decision inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run / "author_coverage.csv").open()))
    checks = json.loads((args.run / "checks.json").read_text())
    cells = defaultdict(list)
    for r in rows:
        cells[(r["world"], r["panel"])].append(r)
    out = []
    for (world, panel), rs in sorted(cells.items()):
        seeds = np.array([r["seed"] for r in rs])
        unreliable = [c for c in checks.get("unreliable_cells", []) if c.endswith(world)]
        for arm in ARMS:
            v = np.array([float(r[arm]) for r in rs])
            lo, hi = boot(v, seeds)
            wl, wh = wilson(round(v.sum() * 3), 3 * len(v))
            out.append({"world": world, "panel": panel, "arm": arm, "authors": len(rs), "coverage": float(v.mean()), "cluster_lo": float(lo), "cluster_hi": float(hi),
                        "wilson_lo": wl, "wilson_hi": wh, "contains_nominal": bool(lo <= 0.95 <= hi),
                        "width": float(np.mean([float(r[f"{arm}_width"]) for r in rs])) if arm != "oracle" else "",
                        "source_var_share": float(np.mean([float(r["source_var_share"]) for r in rs])) if arm == "augmented" else "",
                        "unreliable_bootstrap_cells": ";".join(unreliable) if arm in ("augmented", "mixture") else ""})
        d = np.array([float(r["augmented"]) - float(r["plugin"]) for r in rs])
        lo, hi = boot(d, seeds)
        out.append({"world": world, "panel": panel, "arm": "augmented-plugin", "authors": len(rs), "coverage": float(d.mean()), "cluster_lo": float(lo), "cluster_hi": float(hi)})
    with (args.run / "coverage_summary.csv").open("x", newline="") as f:
        keys = list(out[0])
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o["world"], o["panel"], o["arm"], f"{o['coverage']:.4f} [{o['cluster_lo']:.4f},{o['cluster_hi']:.4f}]", "nominal-in" if o.get("contains_nominal") else "", f"w={o['width']:.3f}" if isinstance(o.get("width"), float) else "", f"srcshare={o['source_var_share']:.3f}" if isinstance(o.get("source_var_share"), float) else "")


if __name__ == "__main__":
    main()
