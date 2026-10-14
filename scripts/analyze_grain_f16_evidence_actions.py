#!/usr/bin/env python3
"""Explain existing T1 edit effects using exact fixed-support Gaussian weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def response_weights(rows, base):
    """Return each frozen source model's affine contrast on unchanged support."""
    x = np.eye(4)[[[-1.0, -1 / 3, 1 / 3, 1.0].index(r["z"]) for r in rows]]
    contrast = np.array([-1.0, 0.0, 0.0, 1.0])
    weights, intercepts = [], []
    for model in base["source_models"]:
        mu, cov = np.asarray(model["mu"]), np.asarray(model["Sigma"])
        observation = x @ cov @ x.T + model["sigma2"] * np.eye(len(rows))
        w = np.linalg.solve(observation, x @ cov @ contrast)
        weights.append(w)
        intercepts.append(float(contrast @ mu - w @ (x @ mu)))
    return np.array(intercepts), np.array(weights)


def analyze(root, output):
    """Reconstruct actual cards; never retune masks, thresholds or model fits."""
    cell = root / "T1"
    bundle = json.loads((cell / "bundle.json").read_text())
    base = bundle["scorer"]
    summaries, weight_rows, hashes = [], [], {}
    for name in ("full_history", "late_offset"):
        path = cell / "cards" / f"{name}.json"
        card = json.loads(path.read_text())
        before = card["before"]["ledger"]["records"]
        after = card["after"]["ledger"]["records"]
        if [{k: v for k, v in r.items() if k != "y"} for r in before] != [
            {k: v for k, v in r.items() if k != "y"} for r in after
        ]:
            raise ValueError("fixed-support weights do not apply after deletions")
        intercepts, weights = response_weights(before, base)
        old_y = np.array([r["y"] for r in before])
        new_y = np.array([r["y"] for r in after])
        change = new_y - old_y
        baseline = intercepts + weights @ old_y
        updated = intercepts + weights @ new_y
        old_parts = card["before"]["result"]["source_components"]
        new_parts = card["after"]["result"]["source_components"]
        actual_old = np.array([p["mean"] for p in old_parts])
        actual_new = np.array([p["mean"] for p in new_parts])
        np.testing.assert_allclose(baseline, actual_old, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(updated, actual_new, atol=1e-10, rtol=1e-8)
        contribution = weights.mean(axis=0) * change
        delta = float(contribution.sum())
        np.testing.assert_allclose(
            delta, card["effect"]["score_delta"], atol=1e-10, rtol=1e-8
        )
        for i, r in enumerate(before):
            weight_rows.append(
                {
                    "case": name,
                    "event_id": r["event_id"],
                    "t": r["t"],
                    "condition": r["z"],
                    "weight": float(weights[:, i].mean()),
                    "value_edit": float(change[i]),
                    "score_contribution": float(contribution[i]),
                }
            )
        absolute = float(np.abs(contribution).sum())
        summaries.append(
            {
                "case": name,
                "source_models": len(weights),
                "max_source_score_error": float(
                    max(
                        np.max(np.abs(baseline - actual_old)),
                        np.max(np.abs(updated - actual_new)),
                    )
                ),
                "exact_weighted_delta": delta,
                "actual_card_delta": card["effect"]["score_delta"],
                "sum_absolute_contributions": absolute,
                "cancellation_fraction": float(1 - abs(delta) / absolute)
                if absolute
                else None,
                "changed_conditions": [
                    r["z"] for i, r in enumerate(before) if change[i] != 0
                ],
            }
        )
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "fixed_support_weights.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    result = {
        "scope": "Post-result algebraic explanation of already frozen cards. Fixed-source, fixed-support T1 only;not a general edit attribution for T2/T3 or record deletion. No new outcome/model fit.",
        "cases": summaries,
        "input_hashes": hashes,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    """Write a separate explanatory artifact without modifying scored cards."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analyze(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
