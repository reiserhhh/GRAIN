#!/usr/bin/env python3
"""Evaluate frozen F04 source models on prefixes and condition-diversity designs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, grouped
from grain_research.gptpro_foundation_v3.f04_recovery import (
    generate_response_world,
    observed_arrays,
    score_response,
    score_source_halves,
)


def table(path, rows):
    """Write computed rows with an explicit union schema."""
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    """Serialize finite computed arrays and scalars."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.tolist(), allow_nan=False)
        + "\n"
    )


def predict_prefix(prefix, models):
    """Return observed-data predictions only; no latent target is accepted."""
    x, y = observed_arrays(prefix)
    target = np.column_stack((np.ones(4), CONDITIONS))
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    individual = target @ beta
    support = np.array([any(row["z"] == z for row in prefix) for z in CONDITIONS])
    if rank < 2:
        individual[~support] = np.nan
    output = {
        "pooled_OLS": target @ np.asarray(models["pooled_OLS"]["beta"]),
        "personal_OLS": individual,
        "REML": score_response(prefix, models["REML"])["means"],
        "four_cell": score_response(prefix, models["four_cell"])["means"],
        "legacy_halves": score_source_halves(prefix, models["legacy_halves"])["means"],
    }
    return output, {
        "rank": int(rank),
        "observed_cells": int(support.sum()),
        "xtx": x.T @ x,
        "slope_data_identified": rank == 2,
    }


def main():
    """Cold-score existing source bundles; do not fit or choose from application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("preserve previous revisions")
    out.mkdir(parents=True)
    (out / "source_archive").mkdir()
    manifest = json.loads((args.source / "output_manifest.json").read_text())
    for entry in manifest["files"]:
        if sha256(args.source / entry["path"]) != entry["sha256"]:
            raise ValueError("source artifact changed")
    source_summary = json.loads((args.source / "summary.json").read_text())
    if bool(source_summary["smoke"]) != args.smoke:
        raise ValueError("smoke/full source mismatch")
    inputs = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f04_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f02_recovery.py",
        ROOT / "grain_research/response_sharing.py",
    ]
    for path in inputs:
        (out / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        out / "input_manifest.json",
        {
            "source_manifest_sha256": sha256(args.source / "output_manifest.json"),
            "files": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        },
    )
    predictions = []
    information = []
    truth_targets = {}
    budgets = [4, 8] if args.smoke else [8, 16, 24, 32]
    for outcome in source_summary["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        models = json.loads((root / "source_models.json").read_text())
        fit = json.loads((root / "fit_manifest.json").read_text())
        assert sha256(root / "source_models.json") == fit["models_sha256"]
        for k in (1, 2, 4):
            if k == 4:
                rows = [
                    json.loads(line)
                    for role in ("validation", "application")
                    for line in (root / f"{role}.jsonl").read_text().splitlines()
                ]
            else:
                # Same author parameters/noise as the stored K4 data. Only the
                # assigned condition design changes; source learning stays fixed.
                rows, _ = generate_response_world(
                    world,
                    seed,
                    counts={**source_summary["counts"], "source": 0},
                    events=source_summary["events"],
                    condition_count=k,
                )
            for author, person in grouped(rows).items():
                for n in budgets:
                    prefix = person[:n]
                    future = person[n:]
                    result, info = predict_prefix(prefix, models)
                    information.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": author,
                            "role": person[0]["role"],
                            "K": k,
                            "n": n,
                            "rank": info["rank"],
                            "I_working": json.dumps(
                                (info["xtx"] / models["REML"]["sigma2"]).tolist()
                            ),
                            "slope_data_identified": info["slope_data_identified"],
                        }
                    )
                    for method, means in result.items():
                        forecast = np.array(
                            [means[list(CONDITIONS).index(row["z"])] for row in future]
                        )
                        future_y = np.array([row["y"] for row in future])
                        predictions.append(
                            {
                                "world": world,
                                "seed": seed,
                                "author_id": author,
                                "role": person[0]["role"],
                                "K": k,
                                "n": n,
                                "method": method,
                                "means": means.tolist(),
                                "future_n": len(future),
                                "future_MSE": float(np.mean((forecast - future_y) ** 2))
                                if len(future)
                                else None,
                                "rank": info["rank"],
                            }
                        )
        # Truth enters only after this world/seed's predictions are made.
        for row in json.loads((root / "truth.json").read_text()):
            truth_targets[(world, seed, row["author_id"])] = row
        print(
            json.dumps({"world": world, "seed": seed, "application_scored": True}),
            flush=True,
        )
    scored = []
    for row in predictions:
        truth = truth_targets[(row["world"], row["seed"], row["author_id"])]
        means = np.asarray(row.pop("means"))
        target = np.asarray(truth["means"])
        complete = bool(np.isfinite(means).all())
        metrics = {
            "future_MSE": row["future_MSE"],
            "profile_MSE": float(np.mean((means - target) ** 2)) if complete else None,
            "delta_squared_error": float(
                ((means[-1] - means[0]) - (target[-1] - target[0])) ** 2
            )
            if complete
            else None,
            "level_squared_error": float((means.mean() - target.mean()) ** 2)
            if complete
            else None,
        }
        legal_linear = row["world"] in {"L", "V", "S", "0", "H"}
        metrics["a_squared_error"] = (
            float((means.mean() - truth["a"]) ** 2)
            if complete and legal_linear
            else None
        )
        metrics["b_squared_error"] = (
            float(((means[-1] - means[0]) / 2 - truth["b"]) ** 2)
            if complete and legal_linear
            else None
        )
        scored.append(
            {
                **row,
                "means": json.dumps(
                    [float(v) if np.isfinite(v) else None for v in means]
                ),
                "complete_profile": complete,
                **metrics,
            }
        )
    table(out / "response_metrics.csv", scored)
    table(out / "diversity_information.csv", information)
    summaries = defaultdict(list)
    for row in scored:
        if row["role"] == "application":
            summaries[(row["world"], row["K"], row["n"], row["method"])].append(row)
    aggregate = []
    for key, rows in summaries.items():
        entry = dict(zip(("world", "K", "n", "method"), key))
        entry["authors"] = len(rows)
        for metric in (
            "future_MSE",
            "profile_MSE",
            "delta_squared_error",
            "level_squared_error",
            "a_squared_error",
            "b_squared_error",
        ):
            values = [row[metric] for row in rows if row[metric] is not None]
            entry[metric] = float(np.mean(values)) if values else None
            entry[metric + "_n"] = len(values)
        aggregate.append(entry)
    table(out / "aggregate_metrics.csv", aggregate)
    contrasts = []
    for world in sorted({row["world"] for row in scored}):
        for method in ("REML", "four_cell", "legacy_halves"):
            for metric in ("future_MSE", "profile_MSE", "delta_squared_error"):
                selected = [
                    row
                    for row in scored
                    if row["world"] == world
                    and row["role"] == "application"
                    and row["K"] == 4
                    and row["n"] == (4 if args.smoke else 24)
                ]
                differences = {}
                for seed in sorted({r["seed"] for r in selected}):
                    a = {
                        r["author_id"]: r[metric]
                        for r in selected
                        if r["seed"] == seed and r["method"] == method
                    }
                    b = {
                        r["author_id"]: r[metric]
                        for r in selected
                        if r["seed"] == seed and r["method"] == "personal_OLS"
                    }
                    assert a.keys() == b.keys()
                    differences[seed] = np.array([a[u] - b[u] for u in sorted(a)])
                contrasts.append(
                    {
                        "world": world,
                        "method": method,
                        "reference": "personal_OLS",
                        "metric": metric,
                        **paired_bootstrap(differences),
                    }
                )
    table(out / "paired_comparisons.csv", contrasts)
    dump(
        out / "summary.json",
        {
            "status": "APPLICATION_COMPARISONS_EXECUTED_TAILS_PENDING",
            "smoke": args.smoke,
            "prediction_rows": len(scored),
            "information_rows": len(information),
            "paired_comparisons": len(contrasts),
            "source_refitted": False,
            "remaining": [
                "V/H variance-tail inference",
                "independent metric audit",
                "original card packaging",
            ],
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(out)), "sha256": sha256(p)}
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ]
        },
    )


if __name__ == "__main__":
    main()
