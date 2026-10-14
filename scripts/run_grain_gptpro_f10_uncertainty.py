#!/usr/bin/env python3
"""Pair/task bootstrap frozen native actions; refit source direct-text contrast."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def resample(rng, indices, pairs, unit):
    """Sample pairs, or tasks then pairs within tasks, retaining paired methods."""
    if unit == "pair":
        return rng.choice(indices, len(indices), replace=True)
    tasks = sorted({pairs[i]["situation"] for i in indices})
    sampled = []
    for task in rng.choice(tasks, len(tasks), replace=True):
        group = [i for i in indices if pairs[i]["situation"] == task]
        sampled.extend(rng.choice(group, len(group), replace=True))
    return np.array(sampled)


def direct_fit(dx, ds, indices):
    """Refit the original source ridge and source-only shrinkage on each sample."""
    scale = dx[indices].std(axis=0)
    scale[scale < 1e-12] = 1
    x = dx[indices] / scale
    coef = np.linalg.lstsq(
        np.vstack([x, np.eye(4)]),
        np.vstack([ds[indices], np.zeros((4, 2))]),
        rcond=None,
    )[0]
    predicted = (dx / scale) @ coef
    grid = np.array([0, 0.25, 0.5, 0.75, 1.0])
    loss = np.mean(
        (grid[:, None, None] * predicted[indices] - ds[indices]) ** 2, axis=1
    )
    choices = grid[np.argmin(loss, axis=0)]
    return predicted * choices


def main():
    """Report loss differences with selection/fit uncertainty correctly scoped."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = list(csv.DictReader((args.cache / "pairs.csv").open()))
    with np.load(args.cache / "action_arrays.npz") as z:
        dx, ds, source = z["dx"], z["ds"], z["source"]
    order = json.loads((args.cache / "pair_order.json").read_text())
    if order != [r["pair"] for r in pairs]:
        raise ValueError("pair ordering")
    native = [
        json.loads(line)
        for line in (args.cache / "native_effects.jsonl").read_text().splitlines()
    ]
    lookup = {(r["pair"], r["query"], r["kind"]): r["effect"] for r in native}
    src, app = np.flatnonzero(source), np.flatnonzero(~source)
    queries = ("extraversion", "neuroticism")
    compiled, donor = np.zeros_like(ds), np.zeros_like(ds)
    for i in app:
        for q, query in enumerate(queries):
            compiled[i, q] = lookup[pairs[i]["pair"], query, "compiled"]
            donor[i, q] = lookup[pairs[i]["pair"], query, "donor"]

    def differences(pred, indices):
        y, c, d, p = ds[indices], compiled[indices], donor[indices], pred[indices]
        return np.array(
            [
                np.mean((p - y) ** 2 - y**2, axis=0),
                np.mean((c - d) ** 2 - d**2, axis=0),
                np.mean((c - y) ** 2 - y**2, axis=0),
                np.mean((d - y) ** 2 - y**2, axis=0),
            ]
        )

    point = differences(direct_fit(dx, ds, src), app)
    names = (
        "direct_full_vs_zero",
        "compiled_donor_vs_zero",
        "compiled_full_vs_zero",
        "donor_full_vs_zero",
    )
    rows = []
    for unit in ("pair", "task_then_pair"):
        rng = np.random.default_rng(20260915)
        samples = []
        for _ in range(2000):
            sampled_source = resample(rng, src, pairs, unit)
            sampled_app = resample(rng, app, pairs, unit)
            samples.append(differences(direct_fit(dx, ds, sampled_source), sampled_app))
        values = np.array(samples)
        np.save(args.output / f"bootstrap_{unit}.npy", values)
        for method, name in enumerate(names):
            for q, query in enumerate(queries):
                low, high = np.quantile(values[:, method, q], [0.025, 0.975])
                rows.append(
                    {
                        "unit": unit,
                        "comparison": name,
                        "query": query,
                        "mse_delta": float(point[method, q]),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "uncertainty": "source_refit_and_application"
                        if method == 0
                        else "fixed_native_fit_application_only",
                    }
                )
    with (args.output / "paired_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "BOOTSTRAP_COMPUTED",
        "draws_per_unit": 2000,
        "interval_rows": len(rows),
        "source_tasks": sorted({pairs[i]["situation"] for i in src}),
        "application_tasks": sorted({pairs[i]["situation"] for i in app}),
        "native_source_uncertainty": "unavailable; native model not refit",
        "scope": "synthetic pair/task resampling, not human population intervals",
        "source_shrinkage": "in-sample source MSE as frozen recovery, not cross-validation",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
