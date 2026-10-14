#!/usr/bin/env python3
"""Dense Gaussian replay of every missing-time state and forecast metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_recovery import dense_posterior


def main():
    """Replay fixed estimates with a covariance solve, not the Kalman runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior audit")
    args.output.mkdir(parents=True)
    hashes = 0
    for directory in (args.source, args.result):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            assert sha256(directory / item["path"]) == item["sha256"]
            hashes += 1
    data = pd.read_csv(args.result / "missing_time_metrics.csv")
    checks = 0
    largest = 0.0

    def check(actual, expected):
        nonlocal checks, largest
        if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8):
            raise AssertionError((actual, expected))
        checks += 1
        largest = max(largest, abs(actual - expected))

    source = json.loads((args.source / "summary.json").read_text())
    for outcome in source["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        model = json.loads((root / "source_model.json").read_text())
        truth = {
            x["author_id"]: x for x in json.loads((root / "truth.json").read_text())
        }
        people = grouped(
            [
                json.loads(line)
                for line in (root / "application.jsonl").read_text().splitlines()
            ]
        )
        predictions = json.loads(
            (args.result / f"{world}_{seed}" / "observed_predictions.json").read_text()
        )
        estimates = {
            (x["author_id"], x["budget"], x["estimator"], x["clock"]): x
            for x in predictions
        }
        for row in data[(data.world == world) & (data.seed == seed)].itertuples(
            index=False
        ):
            target = truth[row.author_id]
            selected = people[row.author_id][: row.budget : 3]
            last = selected[-1]["t"]
            check(row.last_t, last)
            check(row.observations, len(selected))
            if row.clock == "wrong_gap1":
                selected = [{**x, "t": i} for i, x in enumerate(selected)]
            if row.estimator == "oracle":
                psi = [target["a"], target["b"], eta_from_rho(target["rho"])]
                logs = np.log([target["sigma_w"], target["sigma_e"]])
            else:
                psi = estimates[(row.author_id, row.budget, row.estimator, row.clock)][
                    "psi"
                ]
                logs = model["log_sigma"]
            posterior = dense_posterior(psi, selected, logs)
            state = target["states"][last]
            rho = 0.99 * np.tanh(psi[2])
            check(row.state_mean, posterior["state_mean"])
            check(row.state_truth, state)
            check(row.state_MSE, (posterior["state_mean"] - state) ** 2)
            check(row.rho_MSE, (rho - target["rho"]) ** 2)
            check(
                row.state_coverage,
                float(
                    abs(posterior["state_mean"] - state)
                    <= 1.959963984540054 * np.sqrt(posterior["state_variance"])
                ),
            )
            for k in (1, 4):
                future = people[row.author_id][last + k]
                prediction = (
                    psi[0] + psi[1] * future["z"] + rho**k * posterior["state_mean"]
                )
                conditional = (
                    target["a"] + target["b"] * future["z"] + target["rho"] ** k * state
                )
                check(getattr(row, f"forecast{k}_MSE"), (prediction - future["y"]) ** 2)
                check(
                    getattr(row, f"conditional{k}_MSE"), (prediction - conditional) ** 2
                )
    report = {
        "status": "DENSE_REPLAY_PASS",
        "rows": len(data),
        "scalar_checks": checks,
        "max_absolute_error": largest,
        "hashes": hashes,
        "scope": "same-agent covariance implementation replays fixed estimates and actual time anchors; not independent research-team reproduction",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    main()
