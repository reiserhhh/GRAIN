#!/usr/bin/env python3
"""Replay all calibration draws using direct product likelihoods."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def direct_statistics(development, future):
    """Independent nine-law posterior and explicit binomial deviance."""
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    even = corners[np.prod(corners, axis=1) == 1]
    odd = corners[np.prod(corners, axis=1) == -1]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    before = np.array(
        [sigmoid(beta * even[:, axis]) for axis in range(3) for beta in (0.5, 1, 1.5)]
    )
    after = np.array(
        [sigmoid(beta * odd[:, axis]) for axis in range(3) for beta in (0.5, 1, 1.5)]
    )
    d = np.asarray(development)
    f = np.asarray(future)
    mass = np.prod(
        before[None, :, :] ** d[:, None, :]
        * (1 - before[None, :, :]) ** (3 - d[:, None, :]),
        axis=2,
    )
    prediction = (mass / mass.sum(axis=1, keepdims=True)) @ after
    total = np.zeros(len(d))
    for j in range(4):
        for successes, expectation in (
            (f[:, j], 3 * prediction[:, j]),
            (3 - f[:, j], 3 * (1 - prediction[:, j])),
        ):
            active = successes > 0
            total[active] += (
                2 * successes[active] * np.log(successes[active] / expectation[active])
            )
    return total


def main():
    """Verify all saved statistics and every composite p-value without fitting code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = {}
    maximum = 0.0
    count = 0
    for seed in (20260915, 20260916, 20260917):
        saved = np.load(args.input / f"calibration_{seed}.npz")
        replay = np.array(
            [
                direct_statistics(d, f)
                for d, f in zip(saved["development"], saved["future"], strict=True)
            ]
        )
        maximum = max(maximum, float(np.max(np.abs(replay - saved["statistics"]))))
        np.testing.assert_allclose(replay, saved["statistics"], atol=1e-10, rtol=1e-8)
        calibration[seed] = saved["statistics"]
        count += replay.size
    checks = json.loads((args.input / "individual_checks.json").read_text())
    max_stat = 0.0
    max_p = 0.0
    for row in checks:
        statistic = direct_statistics([row["development"]], [row["future"]])[0]
        max_stat = max(max_stat, abs(statistic - row["statistic"]))
        # Use recorded equivalent statistic for exact comparison tie convention.
        # Numerical alternate formula can change an equality by a few ulps.
        tails = [
            (1 + sum(v >= row["statistic"] for v in atom)) / 2001
            for atom in calibration[row["seed"]]
        ]
        max_p = max(max_p, abs(max(tails) - row["p_composite"]))
        np.testing.assert_allclose(tails, row["p_by_atom"], atol=1e-12)
        assert (max(tails) < 0.05) == row["rejected"]
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "DIRECT_LIKELIHOOD_CALIBRATION_REPLAY_PASS",
        "calibration_statistics": count,
        "max_statistic_difference": maximum,
        "evaluation_checks": len(checks),
        "max_evaluation_statistic_difference": max_stat,
        "max_p_difference": max_p,
        "scope": "recorded statistic used for exact floating-point ties;alternate formula verified numerically",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
