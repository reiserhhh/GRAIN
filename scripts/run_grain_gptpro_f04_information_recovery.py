#!/usr/bin/env python3
"""Compare exact F04 OLS information with stored Monte Carlo recovery errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, grouped
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def information(z, sigma2):
    """Return exact linear Gaussian information, retaining deficient rank."""
    x = np.column_stack((np.ones(len(z)), z))
    matrix = x.T @ x / sigma2
    rank = np.linalg.matrix_rank(matrix)
    return matrix, np.linalg.inv(matrix) if rank == 2 else None


def main():
    """Terminal evaluation only: no predictor receives the latent variance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve older runs")
    args.output.mkdir(parents=True)
    for directory in (args.source, args.application):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("input hash mismatch")
    data = pd.read_csv(args.application / "response_metrics.csv", dtype={"world": str})
    data = data[(data.role == "application") & (data.method == "personal_OLS")]
    source = json.loads((args.source / "summary.json").read_text())
    truths = {}
    s_rows = []
    for outcome in source["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        people = {
            p["author_id"]: p for p in json.loads((root / "truth.json").read_text())
        }
        for author, person in people.items():
            truths[(world, seed, author)] = person
        if world != "S":
            continue
        observations = grouped(
            [
                json.loads(line)
                for line in (root / "application.jsonl").read_text().splitlines()
            ]
        )
        for n in (8, 16, 24, 32):
            labels, means, slopes = [], [], []
            for author, events in observations.items():
                prefix = events[:n]
                x = np.array([[1, p["z"]] for p in prefix])
                y = np.array([p["y"] for p in prefix])
                labels.append(people[author]["b"] > 0)
                means.append(y.mean())
                slopes.append(np.linalg.lstsq(x, y, rcond=None)[0][1])
            s_rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "authors": len(labels),
                    "mean_auc": roc_auc_score(labels, means),
                    "slope_auc": roc_auc_score(labels, slopes),
                    "positive_mean": float(np.mean(np.array(means)[labels])),
                    "negative_mean": float(
                        np.mean(np.array(means)[np.logical_not(labels)])
                    ),
                }
            )
    output = []
    for (world, seed, k, n), rows in data.groupby(["world", "seed", "K", "n"]):
        if world not in {"L", "V", "S", "0", "H"}:
            continue
        conditions = {1: [-1.0], 2: [-1.0, 1.0], 4: list(CONDITIONS)}[k]
        z = np.resize(conditions, n)
        sigmas = np.array(
            [
                truths[(world, seed, row.author_id)]["sigma2"]
                for row in rows.itertuples()
            ]
        )
        matrix, covariance = information(z, 1.0)
        entry = {
            "world": world,
            "seed": int(seed),
            "K": int(k),
            "n": int(n),
            "authors": len(rows),
            "rank": int(np.linalg.matrix_rank(matrix)),
            "information_per_unit_inverse_variance": json.dumps(matrix.tolist()),
            "information_interpretation": "Gaussian Fisher; H uses OLS second-moment precision, not t5 likelihood Fisher",
        }
        for j, parameter in enumerate(("a", "b")):
            empirical = (
                float(rows[f"{parameter}_squared_error"].mean())
                if covariance is not None
                else None
            )
            expected = (
                float(covariance[j, j] * sigmas.mean())
                if covariance is not None
                else None
            )
            entry[parameter + "_expected_MSE"] = expected
            entry[parameter + "_empirical_MSE"] = empirical
            entry[parameter + "_expected_RMSE"] = (
                np.sqrt(expected) if expected is not None else None
            )
            entry[parameter + "_empirical_RMSE"] = (
                np.sqrt(empirical) if empirical is not None else None
            )
            entry[parameter + "_MSE_ratio"] = (
                empirical / expected if expected is not None else None
            )
        # K1 identifies the one observed response, not intercept and slope alone.
        entry["observed_cell_mean_expected_MSE"] = (
            float(sigmas.mean() / n) if k == 1 else None
        )
        output.append(entry)
    table(args.output / "information_vs_mc_by_seed.csv", output)
    table(args.output / "S_mean_vs_profile_by_seed.csv", s_rows)
    # Sign of each loss improvement is kept seed-wise, not only pooled.
    all_data = pd.read_csv(
        args.application / "response_metrics.csv", dtype={"world": str}
    )
    selection = all_data[
        (all_data.role == "application") & (all_data.K == 4) & (all_data.n == 24)
    ]
    seed_rows = []
    for (world, seed), rows in selection.groupby(["world", "seed"]):
        losses = rows.groupby("method")[
            ["future_MSE", "profile_MSE", "delta_squared_error"]
        ].mean()
        for method in ("REML", "four_cell", "legacy_halves"):
            for metric in losses.columns:
                delta = float(
                    losses.loc[method, metric] - losses.loc["personal_OLS", metric]
                )
                seed_rows.append(
                    {
                        "world": world,
                        "seed": int(seed),
                        "method": method,
                        "metric": metric,
                        "delta": delta,
                        "lower_loss": delta < 0,
                    }
                )
    table(args.output / "loss_directions_by_seed.csv", seed_rows)
    dump(
        args.output / "summary.json",
        {
            "status": "INFORMATION_AND_S_COMPARISON_COMPUTED",
            "information_rows": len(output),
            "S_rows": len(s_rows),
            "seed_contrasts": len(seed_rows),
            "source_refitted": False,
            "scope": "synthetic finite sample, no personality labels",
            "remaining": "paired-bootstrap audit and original card packaging; H likelihood Fisher not claimed",
        },
    )
    dump(
        args.output / "input_manifest.json",
        {
            "source": sha256(args.source / "output_manifest.json"),
            "application": sha256(args.application / "output_manifest.json"),
            "script_sha256": sha256(Path(__file__)),
        },
    )
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
