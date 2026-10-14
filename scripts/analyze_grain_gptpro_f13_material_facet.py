#!/usr/bin/env python3
"""Material facet of the text-route channel (F13-05 gap; no reader forwards).

Uses the archived F16 text-transport reader scores. For the pilot-fit channel
(NLI qualification arm, as in text transport), compares:
- Jeffreys-Dirichlet channel variance (readings treated as iid, as in F16);
- template-cluster bootstrap variance (resample the 15 pilot templates, each
  contributing one material per pole x qualification cell);
- subtype heterogeneity: channel rows by subject status within each cell,
  and the pilot vs application status mix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"


def cat(row):
    """F16 category alphabet: 2*pole_hat + qualification_hat (NLI arm)."""
    return 2 * int(float(row["pole_readout"]) >= 0.5) + int(float(row["nli_qualification"]) >= 0.5)


def probs(counts):
    """Row-normalised channel with Jeffreys prior, shape (2,2,4)."""
    c = counts + 0.5
    return c / c.sum(axis=2, keepdims=True)


def main():
    """Write material_facet.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader((TT / "reader_scores.csv").open()))
    pilot = [r for r in rows if r["role"] == "pilot" and r["pilot_partition"] == "fit"]
    templates = defaultdict(list)
    for r in pilot:
        templates[r["material_id"].rsplit("_", 2)[0]].append(r)
    names = sorted(templates)
    counts = np.zeros((2, 2, 4))
    for r in pilot:
        counts[int(r["pole"]), int(r["qualification"]), cat(r)] += 1
    stored = json.loads((TT / "channels.json").read_text())["pilot"]["counts"]
    rng = np.random.default_rng(np.random.SeedSequence([20260915, 13, 50]))
    dirichlet = np.stack([np.stack([rng.dirichlet(counts[b, q] + 0.5, 4000) for q in (0, 1)], 1) for b in (0, 1)], 1)
    cluster = []
    for _ in range(4000):
        pick = rng.integers(0, len(names), len(names))
        c = np.zeros((2, 2, 4))
        for i in pick:
            for r in templates[names[i]]:
                c[int(r["pole"]), int(r["qualification"]), cat(r)] += 1
        cluster.append(probs(c))
    cluster = np.array(cluster)
    vd = dirichlet.var(axis=0)
    vc = cluster.var(axis=0)
    informative_rows = (counts > 0).sum(axis=2) >= 2
    mask = informative_rows[:, :, None] & (counts > 0)
    deff = vc[mask] / vd[mask]
    by_status = defaultdict(Counter)
    for r in pilot:
        by_status[(r["pole"], r["qualification"], r["subject_status"])][cat(r)] += 1
    sensitivity = {}
    for b, q in ((0, 0), (1, 1)):
        statuses = sorted({k[2] for k in by_status if k[:2] == (str(b), str(q))})
        err = {}
        for st in statuses:
            c = by_status[(str(b), str(q), st)]
            n = sum(c.values())
            err[st] = 1 - c.get(2 * b + q, 0) / n
        pilot_w = {st: sum(by_status[(str(b), str(q), st)].values()) for st in statuses}
        total = sum(pilot_w.values())
        sensitivity[f"pole{b}_qual{q}"] = {"error_rate_by_status": err, "error_at_pilot_mix": sum(err[s] * pilot_w[s] / total for s in statuses),
                                           "error_at_equal_mix": float(np.mean(list(err.values()))), "error_range_over_mixes": [min(err.values()), max(err.values())]}
    mix = {role: Counter((r["qualification"], r["subject_status"]) for r in rows if r["role"] == role and (role != "pilot" or r["pilot_partition"] == "fit")) for role in ("source", "pilot", "application")}
    out = {
        "pilot_fit_materials": len(pilot),
        "templates": len(names),
        "counts_match_stored_channel": counts.tolist() == stored,
        "design_effect_cluster_over_dirichlet": {"min": float(deff.min()), "median": float(np.median(deff)), "max": float(deff.max()), "cells": int(mask.sum())},
        "misread_sensitivity_to_material_mix": sensitivity,
        "channel_by_status": {f"pole{k[0]}_qual{k[1]}_{k[2]}": dict(sorted(v.items())) for k, v in sorted(by_status.items())},
        "status_mix_by_role": {role: {f"q{k[0]}_{k[1]}": n for k, n in sorted(v.items())} for role, v in mix.items()},
        "interpretation": "design effect on informative cells (rows with >=2 observed categories, nonzero cells) ~1 means Jeffreys-Dirichlet draws already cover template-level material sampling within the pilot mix; subtype-specific error rates show that a different material (status) mix changes the channel: this mix dependence is NOT covered by count uncertainty",
    }
    (args.output / "material_facet.json").write_text(json.dumps(out, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps({"reader_scores": hashlib.sha256((TT / "reader_scores.csv").read_bytes()).hexdigest(), "channels": hashlib.sha256((TT / "channels.json").read_bytes()).hexdigest(), "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
