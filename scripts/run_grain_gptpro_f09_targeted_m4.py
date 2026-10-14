#!/usr/bin/env python3
"""F09 targeted M4 likelihood-ratio contrast, exact composite calibration (protocol v1)."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f09_rivals import (
    BETAS,
    EVEN,
    ODD,
    candidates,
    probability,
)

CHECKS = ROOT / "results/grain_gptpro_foundation_v3/F09/calibration_r1/individual_checks.json"
CORNERS = np.vstack([EVEN, ODD])


def loglik(counts, p):
    """Binomial log-likelihood with 3 trials per corner (counts shape (..., 8))."""
    return (binom.logpmf(counts, 3, p)).sum(axis=-1)


def wilson(k, n, z=1.959963984540054):
    """Wilson 95% interval."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def main():
    """Exact calibration, stored-author evaluation, exact power."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atoms = candidates()["atoms"]
    grid = np.array(list(itertools.product(range(4), repeat=8)))  # 65536 x 8
    in_p = np.array([probability(CORNERS, f, b) for f, b in atoms])  # 9 x 8
    m4_p = np.array([probability(CORNERS, "M4", b) for b in BETAS])  # 3 x 8
    ll_in = np.stack([loglik(grid, p) for p in in_p], axis=1)
    ll_m4 = np.stack([loglik(grid, p) for p in m4_p], axis=1)
    t = ll_m4.max(axis=1) - ll_in.max(axis=1)
    order = np.argsort(-t, kind="stable")
    composite_p = np.zeros(len(grid))
    for a in range(len(atoms)):
        prob = np.exp(ll_in[:, a])
        # P_a(T >= t) for every grid point, ties included
        sorted_t, sorted_prob = t[order], prob[order]
        cum = np.cumsum(sorted_prob)
        # for ties, take the cumulative mass up to the last equal value
        last = np.searchsorted(-sorted_t, -sorted_t, side="right") - 1
        p_at = np.empty(len(grid))
        p_at[order] = cum[last]
        composite_p = np.maximum(composite_p, p_at)
    reject = composite_p <= 0.05
    level_by_atom = {f"{f}:{b}": float(np.exp(ll_in[:, a])[reject].sum()) for a, (f, b) in enumerate(atoms)}
    power = {f"beta{b}": float(np.exp(ll_m4[:, j])[reject].sum()) for j, b in enumerate(BETAS)}
    index = {tuple(row): i for i, row in enumerate(grid)}
    stored = json.loads(CHECKS.read_text())
    groups = {"validation_in_family": [], "application_in_family": [], "application_M4": []}
    archived = {k: [] for k in groups}
    for r in stored:
        key = "application_M4" if r["outside"] else f"{r['role']}_in_family"
        counts = tuple(r["development"] + r["future"])
        groups[key].append(bool(reject[index[counts]]))
        archived[key].append(bool(r["rejected"]))
    rates = {k: {"n": len(v), "targeted_rate": float(np.mean(v)), "targeted_wilson": wilson(sum(v), len(v)),
                 "archived_omnibus_rate": float(np.mean(archived[k]))} for k, v in groups.items()}
    bound = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F09/m4_power_bound_r1/summary.json").read_text())
    out = {"max_level_over_atoms": max(level_by_atom.values()), "level_by_atom": level_by_atom, "exact_power": power,
           "np_per_atom_bound_r3": {f"beta{b}": bound[f"beta{b}:r3:min_atom_bound"]["power"] for b in BETAS},
           "stored_author_rates": rates, "rejection_region_size": int(reject.sum())}
    (args.output / "targeted_m4.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
