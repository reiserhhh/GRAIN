#!/usr/bin/env python3
"""F04 S-world pair-cluster bootstrap sensitivity (census gap for F04-06).

S authors share intercepts in pairs (slopes +/-.5). The stored application
comparisons resampled authors. This recomputes the nine S comparisons (K=4,
n=24, application) with (a) seed-stratified author resampling and (b)
seed-stratified pair-cluster resampling, 2000 draws, seed 20260915. Pair
membership is read from truth.json only at this terminal evaluation step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F04 = ROOT / "results/grain_gptpro_foundation_v3/F04"
SEEDS = ("20260915", "20260916", "20260917")


def boot(values, strata, clusters, rng, draws=2000):
    """Seed-stratified bootstrap resampling whole clusters; returns 95% CI."""
    by = defaultdict(lambda: defaultdict(list))
    for v, s, c in zip(values, strata, clusters, strict=True):
        by[s][c].append(v)
    groups = {s: [np.asarray(v) for v in cs.values()] for s, cs in by.items()}
    out = np.empty(draws)
    for b in range(draws):
        total, count = 0.0, 0
        for s in sorted(groups):
            g = groups[s]
            for i in rng.integers(0, len(g), len(g)):
                total += g[i].sum()
                count += g[i].size
        out[b] = total / count
    return np.quantile(out, [0.025, 0.975]).tolist()


def main():
    """Write pair_cluster_sensitivity.csv and summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pair = {}
    for seed in SEEDS:
        for t in json.loads((F04 / f"source_full_r1/S_{seed}/truth.json").read_text()):
            if t["role"] == "application":
                pair[(seed, t["author_id"])] = f"{seed}:{t['a']:.12f}"
    sizes = defaultdict(int)
    for v in pair.values():
        sizes[v] += 1
    with (F04 / "application_full_r1/response_metrics.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["world"] == "S" and r["role"] == "application" and r["K"] == "4" and r["n"] == "24"]
    with (F04 / "application_full_r1/paired_comparisons.csv").open() as f:
        stored = {(r["method"], r["metric"]): r for r in csv.DictReader(f) if r["world"] == "S"}
    units = defaultdict(dict)
    for r in rows:
        units[(r["seed"], r["author_id"])][r["method"]] = r
    out = []
    for (method, metric), ref in sorted(stored.items()):
        keys = sorted(k for k, v in units.items() if method in v and "personal_OLS" in v and v[method][metric] != "" and v["personal_OLS"][metric] != "")
        diff = [float(units[k][method][metric]) - float(units[k]["personal_OLS"][metric]) for k in keys]
        strata = [k[0] for k in keys]
        author_ci = boot(diff, strata, keys, np.random.default_rng(20260915))
        pair_ci = boot(diff, strata, [pair[k] for k in keys], np.random.default_rng(20260915))

        def decide(lo, hi):
            return "SUPPORTED_LOWER_LOSS" if hi < 0 else ("HIGHER_LOSS" if lo > 0 else "INCONCLUSIVE")

        out.append({
            "method": method, "metric": metric, "n_authors": len(keys),
            "n_pairs": len({pair[k] for k in keys}),
            "delta": float(np.mean(diff)), "stored_delta": float(ref["delta"]),
            "stored_ci_low": float(ref["ci_low"]), "stored_ci_high": float(ref["ci_high"]), "stored_decision": ref["decision"],
            "author_ci_low": author_ci[0], "author_ci_high": author_ci[1], "author_decision": decide(*author_ci),
            "pair_ci_low": pair_ci[0], "pair_ci_high": pair_ci[1], "pair_decision": decide(*pair_ci),
            "width_ratio_pair_over_author": (pair_ci[1] - pair_ci[0]) / (author_ci[1] - author_ci[0]),
        })
    with (args.output / "pair_cluster_sensitivity.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summary = {
        "pair_sizes": dict(sorted({str(k): sum(1 for v in sizes.values() if v == k) for k in set(sizes.values())}.items())),
        "decision_changes": [f"{r['method']}:{r['metric']} {r['stored_decision']}->{r['pair_decision']}" for r in out if r["pair_decision"] != r["stored_decision"]],
        "max_abs_delta_vs_stored": max(abs(r["delta"] - r["stored_delta"]) for r in out),
        "width_ratio_range": [min(r["width_ratio_pair_over_author"] for r in out), max(r["width_ratio_pair_over_author"] for r in out)],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for r in out:
        print(r["method"], r["metric"], f"{r['delta']:+.6f} author[{r['author_ci_low']:+.6f},{r['author_ci_high']:+.6f}] pair[{r['pair_ci_low']:+.6f},{r['pair_ci_high']:+.6f}] {r['stored_decision']}->{r['pair_decision']}")


if __name__ == "__main__":
    main()
