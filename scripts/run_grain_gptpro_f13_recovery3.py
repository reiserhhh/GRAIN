#!/usr/bin/env python3
"""F13 recovery 3: binary next-p uncertainty for B1/B2/M under missingness.

See docs/GRAIN_F13_RECOVERY3_PROTOCOL_V1.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import c2_next_probability, generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_binary_ensemble import (
    EnsembleHistory,
    components,
)
from grain_research.measurement_discovery.estimators import ordered_rows

R = ROOT / "results/grain_gptpro_foundation_v3"
BOOT = R / "F13/binary_source_full_r2"
SEEDS = (20260915, 20260916, 20260917)
COUNTS = {"source": 300, "validation": 100, "application": 200}
MECHANISMS = ("none", "MCAR", "MAR", "MNAR")
CANDS = ("b1", "b2", "m")


def mask(ys, mechanism, rng):
    """Declared binary missingness; returns boolean observed flags."""
    observed, last = [], None
    u = rng.random(len(ys))
    for t, y in enumerate(ys):
        if mechanism == "none":
            p = 0.0
        elif mechanism == "MCAR":
            p = 0.4
        elif mechanism == "MAR":
            p = 0.6 if last == 1 else 0.2
        else:
            p = 0.6 if y == 1 else 0.2
        seen = u[t] >= p
        observed.append(seen)
        if seen:
            last = y
    return observed


def main():
    """Run all worlds x seeds x mechanisms; write author metrics and checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, checks, boot_hashes = [], {"qualified_history_max_diff": 0.0, "bootstrap_counts": {}}, {}
    for wi, world in enumerate(("c2_xor", "c2_iid")):
        law = "xor" if world == "c2_xor" else "iid"
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            records, _ = generate_world(world, seed, counts=COUNTS)
            source = [r for r in records if r["role"] == "source"]
            canonical = fit_observed_source(source, world)
            init0 = np.ones((2, 2))
            for g in ordered_rows(source).values():
                init0[g[0]["observation"], g[1]["observation"]] += 1
            init0 /= init0.sum()
            files = sorted(BOOT.glob(f"{world}_{seed}_b*.json"))
            checks["bootstrap_counts"][f"{world}_{seed}"] = len(files)
            boots = []
            for p in files[:5] if args.smoke else files:
                boot_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
                boots.append(json.loads(p.read_text())["fit"])
            fits = [canonical] + boots
            inits = [init0] + [np.asarray(b["initial_pair"], float) for b in boots]
            apps = [(a, g) for a, g in ordered_rows([r for r in records if r["role"] != "source"]).items() if g[0]["role"] == "application"]
            if args.smoke:
                apps = apps[:6]
            for ai, (author, group) in enumerate(apps):
                ys = [int(r["observation"]) for r in group]
                for mi, mech in enumerate(MECHANISMS):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 13, 30, wi, ai, mi]))
                    seen = mask(ys, mech, rng)
                    ens = EnsembleHistory(fits, inits)
                    ref = QualifiedHistory(canonical, init0, float(np.mean([r["observation"] for r in source])), mode="filtered") if ai < 3 else None
                    acc = {c: {k: [] for k in ("se", "bias", "brier", "cover", "width", "state", "personal", "source", "event")} for c in CANDS}
                    for t, y in enumerate(ys):
                        if t >= 12:
                            values, weights, personal = ens.per_state()
                            comp = components(values, weights, personal)
                            truth = c2_next_probability(ys[t - 2], ys[t - 1], 1, law)
                            if ref is not None:
                                pr = ref.predict()["probabilities"]
                                checks["qualified_history_max_diff"] = max(checks["qualified_history_max_diff"], *(abs(pr[c] - comp["point"][i]) for i, c in enumerate(CANDS)))
                            for i, c in enumerate(CANDS):
                                pvar = comp["state"][i] + comp["personal"][i] + comp["source"][i]
                                half = 1.959963984540054 * np.sqrt(pvar)
                                lo, hi = max(0.0, comp["point"][i] - half), min(1.0, comp["point"][i] + half)
                                a = acc[c]
                                a["se"].append((comp["point"][i] - truth) ** 2)
                                a["bias"].append(comp["point"][i] - truth)
                                a["brier"].append((comp["point"][i] - y) ** 2)
                                a["cover"].append(float(lo <= truth <= hi))
                                a["width"].append(hi - lo)
                                for k in ("state", "personal", "source", "event"):
                                    a[k].append(comp[k][i])
                        q = 1.0 if seen[t] else 0.0
                        pole = float(y) if seen[t] else 0.5
                        ens.observe(pole, q)
                        if ref is not None:
                            ref.observe(pole, q)
                    for c in CANDS:
                        rows.append({"world": world, "seed": seed, "author_id": author, "mechanism": mech, "candidate": c,
                                     "missing_rate": 1 - float(np.mean(seen)),
                                     **{k: float(np.mean(v)) for k, v in acc[c].items()}})
            print(json.dumps({"world": world, "seed": seed, "authors": len(apps)}), flush=True)
    with (args.output / "author_metrics.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # replay check (a): no-mask Brier equals end_to_end complete oracle arm
    e2e = {}
    with (R / "F08/end_to_end_r1/metrics_by_author.csv").open() as f:
        for r in csv.DictReader(f):
            if r["reader"] == "lexical" and r["panel"] == "complete" and r["arm"] == "oracle_channel_oracle_source":
                e2e[(r["world"], r["seed"], r["author_id"], r["candidate"])] = float(r["brier"])
    diffs = [abs(r["brier"] - e2e[(r["world"], str(r["seed"]), r["author_id"], r["candidate"])]) for r in rows if r["mechanism"] == "none"]
    checks["no_mask_brier_vs_end_to_end_complete_max_diff"] = max(diffs)
    checks["no_mask_rows_compared"] = len(diffs)
    (args.output / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps({
        "code": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("scripts/run_grain_gptpro_f13_recovery3.py", "grain_research/gptpro_foundation_v3/f13_binary_ensemble.py", "docs/GRAIN_F13_RECOVERY3_PROTOCOL_V1.md")},
        "bootstrap_files": boot_hashes}, indent=1) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
