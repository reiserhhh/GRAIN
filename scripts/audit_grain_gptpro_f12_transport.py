#!/usr/bin/env python3
"""Replay pilot calibration and per-author transport metrics with alternate algebra."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    """Verify source-only coefficients, individual slopes and paired loss intervals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    error = 0.0
    count = 0
    for path in sorted(args.input.glob("calibration_*.json")):
        model = json.loads(path.read_text())
        if model["status"] == "MISSING_ANCHOR":
            assert model["rank"] < model["required_rank"]
            continue
        seed, world, mode = path.stem.removeprefix("calibration_").split("_", 2)
        with np.load(args.input / f"observations_{seed}_pilot.npz") as z:
            old = z["old"][: config["pilot_fit"]].ravel()
            condition = np.tile(z["conditions"], config["pilot_fit"])
        if world == "identity":
            new = old
        elif world == "global":
            new = 1 + 1.5 * old
        elif world == "conditional":
            new = condition + (1 + 0.25 * condition) * old
        else:
            new = old + 0.1 * old**3
        if mode == "global":
            design = np.column_stack([np.ones(len(old)), old])
            target = new
        elif mode == "conditional":
            design = np.column_stack(
                [np.ones(len(old)), condition, old, condition * old]
            )
            target = new
        else:
            design = np.column_stack([np.ones(len(old)), new, new**3])
            target = old
        q, r = np.linalg.qr(design, mode="reduced")
        coef = np.linalg.solve(r, q.T @ target)
        error = max(error, float(np.max(np.abs(coef - model["coefficients"]))))
        if mode == "cubic":
            points = [min(new), max(new)] + ([0.0] if min(new) <= 0 <= max(new) else [])
            minimum = min(coef[1] + 3 * coef[2] * p * p for p in points)
            assert (minimum <= 0) == (model["status"] == "NONMONOTONE_INVERSE")
        count += 1
    with (args.input / "author_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cache = {}
    for row in rows:
        key = tuple(row[k] for k in ("seed", "world", "role", "method"))
        if key not in cache:
            with np.load(args.input / f"scores_{'_'.join(key)}.npz") as z:
                cache[key] = (z["score"], z["support"])
        score, support = cache[key]
        i = int(row["index"])
        assert (row["supported"] == "True") == bool(support[i, :24].all())
        if row["supported"] != "True":
            continue
        z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 6)
        intercept = score[i, :24].mean()
        slope = np.dot(score[i, :24], z) / np.dot(z, z)
        error = max(error, abs(2 * slope - float(row["delta_prediction"])))
        role = "pilot" if row["role"] == "pilot_validation" else "application"
        with np.load(args.input / f"truth_{row['seed']}_{role}.npz") as truth:
            theta = truth["theta"][i + (config["pilot_fit"] if role == "pilot" else 0)]
        mse = np.mean(((intercept - theta[0]) + (slope - theta[1]) * z[:4]) ** 2)
        error = max(
            error,
            abs(mse - float(row["cell_mse"])),
            abs(2 * theta[1] - float(row["delta_truth"])),
        )
    intervals = []
    for world, method, reference in (
        ("global", "global", "uncalibrated"),
        ("conditional", "conditional", "global"),
        ("nonlinear", "global", "uncalibrated"),
    ):
        groups = []
        for seed in config["seeds"]:
            subset = [
                r
                for r in rows
                if r["seed"] == str(seed)
                and r["world"] == world
                and r["role"] == "application"
            ]
            a = {r["index"]: r for r in subset if r["method"] == method}
            b = {r["index"]: r for r in subset if r["method"] == reference}
            groups.append(
                np.array(
                    [
                        (float(a[i]["delta_prediction"]) - float(a[i]["delta_truth"]))
                        ** 2
                        - (float(b[i]["delta_prediction"]) - float(b[i]["delta_truth"]))
                        ** 2
                        for i in a
                    ]
                )
            )
        rng = np.random.default_rng(20260915)
        samples = [
            np.mean(
                np.concatenate([rng.choice(d, len(d), replace=True) for d in groups])
            )
            for _ in range(2000)
        ]
        lo, hi = np.quantile(samples, [0.025, 0.975])
        intervals.append(
            {
                "world": world,
                "method": method,
                "reference": reference,
                "delta_mse_difference": float(np.mean(groups)),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    if error > 1e-9:
        raise ValueError("transport numerical replay failed")
    result = {
        "status": "PILOT_QR_AND_AUTHOR_REPLAY_PASS",
        "calibrations": count,
        "author_rows": len(rows),
        "max_error": error,
        "paired_intervals": intervals,
        "interval_scope": "fixed pilot/source fits,application authors stratified by seed",
        "remaining": "full score/support table and aggregate rank/coverage replay;signature identity;source-fit uncertainty",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
