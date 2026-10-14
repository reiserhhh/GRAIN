#!/usr/bin/env python3
"""Rebuild MTMM covariance cell-by-cell and audit all cold predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def covariance(s, method, event, error, conditional):
    """Independent scalar covariance definition for all crossed cell pairs."""
    cells = [(k, r, t) for k in range(3) for r in range(3) for t in range(2)]
    v, cross = np.zeros((18, 18)), np.zeros((3, 18))
    for i, (k, r, t) in enumerate(cells):
        loading = 1 + 0.3 * (r - 1) * (2 * t - 1) if conditional else 1
        cross[:, i] = s[:, k] * loading
        for j, (kk, rr, tt) in enumerate(cells):
            other = 1 + 0.3 * (rr - 1) * (2 * tt - 1) if conditional else 1
            v[i, j] = (
                loading * other * s[k, kk]
                + method * (r == rr)
                + event * (t == tt)
                + error * (i == j)
            )
    return v, cross


def main():
    """Audit source REML objectives, latent construction, scores and variances."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    error, objective_error, rows = 0.0, 0.0, 0
    for path in sorted(args.input.glob("fit_*.json")):
        seed, rest = path.stem.removeprefix("fit_").split("_", 1)
        world, name = rest.rsplit("_", 1)
        model = json.loads(path.read_text())
        s, mean = np.array(model["sigma"]), np.array(model["mean"])
        v, cross = covariance(
            s,
            model["method"],
            model["event"],
            model["error"],
            world == "conditional_loading",
        )
        with np.load(args.input / f"data_{seed}_{world}_source.npz") as z:
            source = z["observed"]
        if name == "reml":
            centered = source - source.mean(axis=0)
            inverse = np.linalg.inv(v)
            objective = 0.5 * (
                (len(source) - 1) * np.linalg.slogdet(v)[1]
                + np.sum((centered @ inverse) * centered)
            )
            objective_error = max(objective_error, abs(objective - model["objective"]))
        for role in ("validation", "application"):
            with np.load(args.input / f"data_{seed}_{world}_{role}.npz") as z:
                y = z["observed"]
            for panel, cells in (
                ("all_two_occasions", np.arange(18)),
                ("same_event_three_readers", np.arange(0, 18, 2)),
                ("one_reader_two_occasions", np.array([0, 1, 6, 7, 12, 13])),
            ):
                selected = cross[:, cells]
                inv = np.linalg.inv(v[np.ix_(cells, cells)])
                pred = (y[:, cells] - mean[cells]) @ inv @ selected.T
                variance = s - selected @ inv @ selected.T
                with np.load(
                    args.input / f"pred_{seed}_{world}_{name}_{role}_{panel}.npz"
                ) as z:
                    error = max(
                        error,
                        float(np.max(np.abs(pred - z["prediction"]))),
                        float(np.max(np.abs(variance - z["variance"]))),
                    )
                rows += pred.size
    # Two distinct constant-author decompositions give exactly the same y law.
    s = np.full((3, 3), 0.3)
    np.fill_diagonal(s, 1.0)
    v1, cross1 = covariance(s, 0.25, 0, 0.25, False)
    v2, cross2 = covariance(s + 0.25 * np.ones((3, 3)), 0.25, 0, 0.25, False)
    equivalence_error = float(np.max(np.abs(v1 + 0.25 * np.ones((18, 18)) - v2)))
    anchor_difference = float(np.max(np.abs(cross1 - cross2)))
    if error > 1e-10 or objective_error > 1e-8 or equivalence_error > 1e-12:
        raise ValueError("MTMM arithmetic audit failed")
    result = {
        "status": "PASS_ALTERNATE_NUMERICAL_REPLAY",
        "prediction_scalars": rows,
        "max_prediction_variance_error": error,
        "max_reml_objective_error": objective_error,
        "constant_method_equivalence_error": equivalence_error,
        "independent_anchor_crosscov_difference": anchor_difference,
        "independent_agent": False,
        "remaining": "optimizer optimality and aggregate metrics/interval audit remain; no human validity",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
