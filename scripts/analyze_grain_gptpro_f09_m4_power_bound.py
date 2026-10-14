#!/usr/bin/env python3
"""F09-06 M4 power/budget boundary: exact Neyman-Pearson upper bounds.

The calibrated out-of-family check observes r responses at each of the 8
corners (design: r=3, 24 events). Any test with level alpha under a simple
in-family atom has power against M4(beta) at most the randomized NP power for
that atom; a test that is level alpha at every atom is bounded by the minimum
over atoms. A test calibrated only on the prior mixture of atoms is bounded by
the NP power against the mixture law (computed exactly for r <= 6). Exact
enumeration over sufficient count vectors; no simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f09_rivals import (
    BETAS,
    CORNERS,
    candidates,
    probability,
)

ALPHA = 0.05


def np_power(log_p0, log_p1, alpha=ALPHA):
    """Randomized NP power from joint log-pmfs over the same support."""
    llr = log_p1 - log_p0
    order = np.argsort(-llr, kind="stable")
    p0 = np.exp(log_p0[order])
    p1 = np.exp(log_p1[order])
    llr = llr[order]
    power = size = 0.0
    i = 0
    while i < len(llr):
        j = i
        while j < len(llr) and llr[j] == llr[i]:
            j += 1
        block0, block1 = p0[i:j].sum(), p1[i:j].sum()
        if size + block0 <= alpha:
            size += block0
            power += block1
        else:
            power += block1 * (alpha - size) / block0 if block0 > 0 else block1
            break
        i = j
    return float(power)


def grouped_logpmf(r, law_rows):
    """Enumerate count vectors over corner groups with identical probabilities."""
    keys = [tuple(np.round(col, 12)) for col in np.asarray(law_rows).T]
    groups = {}
    for c, k in enumerate(keys):
        groups.setdefault(k, []).append(c)
    sizes = [len(v) for v in groups.values()]
    probs = np.array([list(k) for k in groups])  # (groups, laws)
    grids = np.meshgrid(*[np.arange(r * m + 1) for m in sizes], indexing="ij")
    counts = np.stack([g.ravel() for g in grids], axis=1)
    out = []
    for law in range(probs.shape[1]):
        lp = np.zeros(len(counts))
        for g, m in enumerate(sizes):
            lp += binom.logpmf(counts[:, g], r * m, probs[g, law])
        out.append(lp)
    return out


def main():
    """Write power_bounds.csv and summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atoms = candidates()["atoms"]
    prior = candidates()["prior"]
    rows = []
    for beta_m4 in BETAS:
        p1 = probability(CORNERS, "M4", beta_m4)
        for family, beta in atoms:
            p0 = probability(CORNERS, family, beta)
            kl = float(np.mean(p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))))
            for r in (3, 6, 12, 24):
                if r >= 24 and len({tuple(np.round((a, b), 12)) for a, b in zip(p0, p1, strict=True)}) > 4:
                    power = ""
                else:
                    lp0, lp1 = grouped_logpmf(r, [p0, p1])
                    power = np_power(lp0, lp1)
                rows.append({"m4_beta": beta_m4, "null": f"{family}:{beta}", "reps_per_corner": r, "events": 8 * r, "kl_per_event": kl, "np_power_bound": power})
        for r in (3, 6):
            laws = [probability(CORNERS, f, b) for f, b in atoms] + [p1]
            lps = grouped_logpmf(r, laws)
            mix = np.logaddexp.reduce(np.stack([np.log(w) + lp for w, lp in zip(prior, lps[:-1], strict=True)]), axis=0)
            rows.append({"m4_beta": beta_m4, "null": "prior_mixture", "reps_per_corner": r, "events": 8 * r, "kl_per_event": "", "np_power_bound": np_power(mix, lps[-1])})
    with (args.output / "power_bounds.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for beta_m4 in BETAS:
        for r in (3, 6, 12, 24):
            vals = [x for x in rows if x["m4_beta"] == beta_m4 and x["reps_per_corner"] == r and x["null"] != "prior_mixture" and x["np_power_bound"] != ""]
            if vals:
                best = min(vals, key=lambda x: x["np_power_bound"])
                summary[f"beta{beta_m4}:r{r}:min_atom_bound"] = {"power": best["np_power_bound"], "nearest_null": best["null"], "atoms_evaluated": len(vals)}
        for r in (3, 6):
            mix = next(x for x in rows if x["m4_beta"] == beta_m4 and x["reps_per_corner"] == r and x["null"] == "prior_mixture")
            summary[f"beta{beta_m4}:r{r}:mixture_bound"] = mix["np_power_bound"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
