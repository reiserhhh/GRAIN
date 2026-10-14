#!/usr/bin/env python3
"""Resolve matched-stranger chance baselines without changing frozen HM94 fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_grain_opportunity_current_hm94 as R

PLAN = ROOT / "docs/GRAIN_OPPORTUNITY_MATCHED_NULL_HM94.md"
TOL = 1e-12


def win(negative, positive):
    """Count superiority with a predeclared numerical all-tie tolerance."""
    delta = negative - positive
    return (delta > TOL).astype(float) + 0.5 * (abs(delta) <= TOL)


def analyze(out):
    """Compare identity association against the actual nuisance-matched null."""
    cfg = R.sealed(out)
    for stage in ("01", "02", "03"):
        R.previous(out, stage)
    target = out / "04_matched_null"
    target.mkdir(exist_ok=False)
    R.write(
        target / "manifest.json",
        {
            "after_primary_outcome": True,
            "before_companion_outcome": True,
            "plan_hash": R.sha(PLAN),
            "code_hash": R.sha(Path(__file__)),
            "input_hash": R.sha(out / "02/private_author_accounts.npz"),
            "draws": cfg["permutation_draws"],
            "seed": cfg["seed"],
            "tie_tolerance": TOL,
        },
    )
    meta = R.read(out / "02/private_metadata.json")
    with np.load(out / "02/private_author_accounts.npz", allow_pickle=False) as z:
        parts = np.stack([z[r["key"] + "_parts"] for r in meta])
        q = np.stack([z[r["key"] + "_q"] for r in meta]).sum(3)
    n = len(meta)
    battery = np.array([len(r["queries"]) for r in meta])
    rng = np.random.default_rng(cfg["seed"])
    permutations = np.tile(np.arange(n), (cfg["permutation_draws"], 1))
    for row in permutations:
        for value in (4, 5):
            ix = np.flatnonzero(battery == value)
            row[ix] = rng.permutation(ix)
    if any(not np.array_equal(battery[row], battery) for row in permutations):
        raise AssertionError("Battery-crossing permutation")
    cycle = parts[..., 2, :, :]
    edges = np.triu_indices(cfg["states"], 1)
    output, null_arrays = [], {}
    for ai, arm in enumerate(cfg["arms"]):
        for ci, channel in enumerate(("raw", "opportunity", "response")):
            feature = cycle[:, :, ai, ci, edges[0], edges[1]]
            degenerate = (
                float(np.ptp(feature.reshape(-1, feature.shape[-1]), axis=0).max())
                < TOL
            )
            observed, unselected, null = [], [], []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                distances = np.sum((feature[:, a, None] - feature[None, :, b]) ** 2, -1)
                context = np.sum((q[:, a, None] - q[None, :, b]) ** 2, -1)
                nearest = np.argsort(context, axis=1, kind="stable")[
                    :, : cfg["matched_strangers"] + 1
                ]
                scores = []
                for permutation in np.vstack((np.arange(n), permutations)):
                    positives = distances[np.arange(n), permutation]
                    eligible = nearest != permutation[:, None]
                    ranks = np.cumsum(eligible, axis=1)
                    selected = eligible & (ranks <= cfg["matched_strangers"])
                    negatives = np.take_along_axis(distances, nearest, 1)[
                        selected
                    ].reshape(n, -1)
                    values = (
                        np.full(n, 0.5)
                        if degenerate
                        else win(negatives, positives[:, None]).mean(1)
                    )
                    scores.append(values)
                observed.append(scores[0])
                null.append(np.array(scores[1:]).mean(1))
                same_battery = battery[:, None] == battery[None]
                np.fill_diagonal(same_battery, False)
                all_wins = win(distances, np.diag(distances)[:, None])
                uu = (all_wins * same_battery).sum(1) / same_battery.sum(1)
                unselected.append(np.full(n, 0.5) if degenerate else uu)
            observed = np.array(observed).mean(0)
            unselected = np.array(unselected).mean(0)
            null = np.array(null).mean(0)
            value = float(observed.mean())
            lower, upper = map(float, np.quantile(null, (0.025, 0.975)))
            null_arrays[f"{arm}_{channel}"] = null
            output.append(
                {
                    "arm": arm,
                    "channel": channel,
                    "degenerate": degenerate,
                    "matched_observed": value,
                    "matched_null_mean": float(null.mean()),
                    "matched_null_lower": lower,
                    "matched_null_upper": upper,
                    "delta_over_matched_null": float(value - null.mean()),
                    "permutation_p": float(
                        (1 + np.sum(null >= value)) / (len(null) + 1)
                    ),
                    "unselected_within_battery_auc": float(unselected.mean()),
                    "unselected_ci_lower": R.M.bootstrap_mean(
                        unselected, cfg["bootstrap_draws"], cfg["seed"]
                    )[1],
                    "unselected_ci_upper": R.M.bootstrap_mean(
                        unselected, cfg["bootstrap_draws"], cfg["seed"]
                    )[2],
                }
            )
    table = pd.DataFrame(output)
    table.to_csv(target / "matched_null_and_unselected.csv", index=False)
    np.savez_compressed(
        target / "null_distributions.npz", **null_arrays, permutations=permutations
    )
    R.finish(
        target,
        {
            "status": "FINITE_HM94_MATCHED_NULL_COMPLETE",
            "permutations": len(permutations),
            "application_authors": n,
            "new_fits": 0,
            "new_calls": 0,
            "constant_accounts": int(table.degenerate.sum()),
            "warning": "Nuisance-matched chance is not universally0.5; null range is not a populationCI",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.OUT)
    analyze(parser.parse_args().out)
