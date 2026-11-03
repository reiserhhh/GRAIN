#!/usr/bin/env python3
"""Apply all old partial-evidence weights to a new observed metadata channel once."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_behavior_pushforward as P
from grain_research.construction.math_conditional_memory import CONTRACTS
from scripts import run_grain_behavior_pushforward_hm291 as R


def main():
    """Read sealed weights,never recompute integration or change source calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    manifest = R.check(args.parent, ("computed_seal.json", "verification_seal.json"))
    cached_parent = ROOT / manifest["config"]["parent"]
    R.check(
        cached_parent,
        ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    out = args.parent / "postoutcome_channels_r1"
    out.mkdir(exist_ok=False)
    source_path = cached_parent / "trajectories.npz"
    R.write(
        out / "manifest.json",
        {
            "classification": "POSTOUTCOME new-endpoint application of old contracts,exploratory",
            "code": {
                p: R.digest(ROOT / p)
                for p in [
                    "scripts/describe_grain_behavior_channels_hm291.py",
                    "docs/GRAIN_BEHAVIOR_CHANNEL_FOLLOWUP_HM291.md",
                ]
            },
            "inputs": {
                str(source_path): R.digest(source_path),
                **{
                    str(args.parent / p): R.digest(args.parent / p)
                    for p in ["observations.npz", "predictions.npz"]
                },
            },
            "all_contracts": list(CONTRACTS),
            "realizations": [0, 1],
            "compute_seconds_cap": 180,
            "old_monotonicity_flags_not_repaired": True,
        },
    )
    z, primary = (
        R.load(args.parent / "observations.npz"),
        R.load(args.parent / "predictions.npz"),
    )
    start, results, replay = time.monotonic(), {}, []
    with np.load(source_path, allow_pickle=False) as cache:
        for ref in manifest["config"]["references"]:
            bundle = R.load(args.parent / f"source_{ref}.npz")
            mu, var = bundle["behavior"].mean(0), bundle["behavior"].var(0)
            tables = [
                P.profile_table(bundle["behavior"], c)[0] for c in bundle["codes"]
            ]
            membership = [np.eye(32)[c] for c in bundle["codes"]]
            for rep in (0, 1):
                weights = np.exp(cache[f"{ref}/rep{rep}/early_logweights"])
                if not np.allclose(weights.sum(-1), 1):
                    raise ArithmeticError("Old source weights not normalized")
                predictions = [weights @ bundle["behavior"]] + [
                    (weights @ m) @ t for m, t in zip(membership, tables, strict=True)
                ]
                gains_by_model = {}
                cells = {}
                for model, pred in zip(R.MODELS, predictions, strict=True):
                    gains = {}
                    for j, contract in enumerate(CONTRACTS):
                        gain = P.risk_gain(
                            z["actual_later_behavior"], pred[:, j], mu, var
                        )
                        foreign = P.risk_gain(
                            z["actual_later_behavior"], pred[z["donors"], j], mu, var
                        )
                        gains[contract] = gain
                        cells[f"{model}/{contract}"] = {
                            "gain_vs_prior": float(gain.mean()),
                            "own_stranger": float((gain - foreign).mean()),
                            "all_feature_gain": gain.mean(0).tolist(),
                            "all_feature_own_stranger": (gain - foreign)
                            .mean(0)
                            .tolist(),
                        }
                        if contract == "full":
                            replay.append(
                                float(
                                    np.max(abs(pred[:, j] - primary[f"{ref}_{model}"]))
                                )
                            )
                        if contract == "none":
                            replay.append(float(np.max(abs(gain))))
                    gains_by_model[model] = gains
                cells["full_minus_condition"] = {
                    name: float((g["full"] - g["condition"]).mean())
                    for name, g in gains_by_model.items()
                }
                cells["full_minus_expression"] = {
                    name: float((g["full"] - g["expression"]).mean())
                    for name, g in gains_by_model.items()
                }
                results[f"{ref}/rep{rep}"] = cells
    elapsed = time.monotonic() - start
    if elapsed > 180 or max(replay) > 1e-10:
        raise ArithmeticError("Budget/full/neutral replay failed;no rescue")
    summary = {
        "results": results,
        "seconds": elapsed,
        "full_neutral_replay_error": max(replay),
        "actual_people": 120,
        "no_new_capture_fit_draws_neural_calls": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(out, "seal.json", ["manifest.json", "summary.json"])
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "replay": max(replay),
                "pooled_full_mode": {
                    r: {
                        c: results[f"pooled/rep{r}"][f"full_mode/{c}"]
                        for c in CONTRACTS
                    }
                    for r in (0, 1)
                },
            }
        )
    )


if __name__ == "__main__":
    main()
