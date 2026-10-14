#!/usr/bin/env python3
"""F06 state-target failure attribution (declared attribution arms, zero world).

Frozen before running (2026-09-26, Claude Opus 5.5). In the nonreactive world
(true beta = 0, physical displacement exactly 0) the state-targeted information
policy had state MSE .395 vs .187 for the fixed alternating design. Two declared
attribution arms isolate the mechanism; they are diagnostics, not new candidate
policies, and nothing is tuned:
- state_z_only: the frozen state criterion chooses z, with h forced to 0
  (isolates condition selection by the criterion);
- fixed_all_h: the fixed alternating z design with h=1 at every step
  (isolates recorded actions interacting with the working beta prior).
Replay arms fixed and scalar_state must equal the archived
F06/gaussian_full_r1 application metrics exactly (budget 24, zero world).
Same author parameters, innovations, noise and random streams as the archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f06_gaussian_design import (
    CANDIDATES,
    TARGETS,
    expected_variance_gain,
    fit_prefix,
)

SOURCE = ROOT / "results/grain_gptpro_foundation_v3/F05/source_full_r2"
ARCHIVE = ROOT / "results/grain_gptpro_foundation_v3/F06/gaussian_full_r1/metrics.csv"
SEEDS = (20260915, 20260916, 20260917)
EVENTS = 24


def state_choice(posterior, allow_h):
    """Frozen state criterion; tie rule identical to policy_action."""
    cands = [c for c in CANDIDATES if allow_h or c[1] == 0.0]
    gains = np.array([expected_variance_gain(posterior, z, h, "state")["gain"] for z, h in cands])
    return cands[int(np.flatnonzero(np.isclose(gains, gains.max(), atol=1e-14, rtol=0))[0])]


def run(seed, ordinal, source, arm):
    """One application author, zero world, one arm; returns state MSE at t=24."""
    base = [seed, 6, 1, 2, ordinal]
    rng = np.random.default_rng(np.random.SeedSequence(base + [0]))
    a, b, rho, _beta = rng.normal(), rng.normal(0.5, 0.5), rng.uniform(0.2, 0.8), rng.normal(0.4, 0.2)
    state = np.random.default_rng(np.random.SeedSequence(base + [1])).normal(0, 0.5 / np.sqrt(1 - rho**2))
    innovations = np.random.default_rng(np.random.SeedSequence(base + [2])).normal(0, 0.5, EVENTS)
    measurement = np.random.default_rng(np.random.SeedSequence(base + [3])).normal(0, 0.5, EVENTS)
    prior = fit_prefix([], source)
    if arm in ("scalar_state", "state_z_only"):
        gains = [expected_variance_gain(prior, z, 0.0, "state")["gain"] for z in (-1.0, 1.0)]
        z = (-1.0, 1.0)[int(np.flatnonzero(np.isclose(gains, max(gains), atol=1e-14, rtol=0))[0])]
    else:
        z = -1.0
    rows, actions = [], 0
    for t in range(EVENTS):
        state = rho * state + 0.0 + innovations[t]
        y = a + b * z + state + measurement[t]
        rows.append({"author_id": f"attr_{seed}_{ordinal}", "event_id": f"{t}", "role": "application", "t": t, "z": float(z), "y": float(y), "h": 0})
        posterior = fit_prefix(rows, source)
        if t + 1 < EVENTS:
            if arm == "fixed":
                next_z, h = (-1.0 if (t + 1) % 2 == 0 else 1.0), 0.0
            elif arm == "fixed_all_h":
                next_z, h = (-1.0 if (t + 1) % 2 == 0 else 1.0), 1.0
            elif arm == "scalar_state":
                next_z, h = state_choice(posterior, True)
            else:
                next_z, h = state_choice(posterior, False)
        else:
            next_z, h = z, 0.0
        rows[-1]["h"] = int(h)
        actions += int(h)
        z = next_z
    idx = TARGETS["state"]
    est, var = posterior["mean"][idx], posterior["covariance"][idx, idx]
    return {"seed": seed, "ordinal": ordinal, "arm": arm, "MSE": float((est - state) ** 2), "bias": float(est - state),
            "covered": float(abs(est - state) <= 1.959963984540054 * np.sqrt(max(var, 0))), "actions": actions,
            "beta_estimate": float(posterior["mean"][TARGETS["beta"]])}


def main():
    """Run arms for 600 application authors; replay-check against the archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authors", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in SEEDS:
        source = json.loads((SOURCE / f"core_{seed}" / "source_model.json").read_text())
        for ordinal in range(args.authors):
            for arm in ("fixed", "scalar_state", "state_z_only", "fixed_all_h"):
                rows.append(run(seed, ordinal, source, arm))
        print(json.dumps({"seed": seed}), flush=True)
    with (args.output / "author_state_metrics.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    archived = {}
    with ARCHIVE.open() as f:
        for r in csv.DictReader(f):
            if r["role"] == "application" and r["reactivity"] == "zero" and r["target"] == "state" and r["budget"] == "24" and r["arm"] in ("fixed", "scalar_state"):
                archived[(r["seed"], r["arm"])] = archived.get((r["seed"], r["arm"]), []) + [float(r["MSE"])]
    replay = {}
    for seed in SEEDS:
        for arm in ("fixed", "scalar_state"):
            mine = [r["MSE"] for r in rows if r["seed"] == seed and r["arm"] == arm]
            arch = archived.get((str(seed), arm), [])
            replay[f"{seed}:{arm}"] = {"n": len(mine), "archived_n": len(arch), "mean_mine": float(np.mean(mine)), "mean_archived": float(np.mean(arch)) if arch else None,
                                       "sorted_max_abs_diff": float(np.max(np.abs(np.sort(mine) - np.sort(arch)))) if len(arch) == len(mine) else None}
    summary = {arm: {k: float(np.mean([r[k] for r in rows if r["arm"] == arm])) for k in ("MSE", "bias", "covered", "actions", "beta_estimate")} for arm in ("fixed", "scalar_state", "state_z_only", "fixed_all_h")}
    (args.output / "summary.json").write_text(json.dumps({"replay": replay, "arms": summary}, indent=1) + "\n")
    print(json.dumps({"replay": replay, "arms": summary}, indent=1))


if __name__ == "__main__":
    main()
