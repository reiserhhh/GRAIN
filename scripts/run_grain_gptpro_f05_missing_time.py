#!/usr/bin/env python3
"""Paired F05 every-third-observation time-unit intervention."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_recovery import score_prefix
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def paired_predictions(events, model, budget, personal):
    """Same observations, two clock conventions; no outcomes after cutoff read."""
    selected = events[:budget:3]
    last = selected[-1]["t"]
    conditions = [-1.0 if ((last + k) // 4) % 2 == 0 else 1.0 for k in range(1, 5)]
    compressed = [{**row, "t": i} for i, row in enumerate(selected)]
    return (
        {
            "correct_gap3": score_prefix(
                selected, model, conditions, fit_personal=personal
            ),
            "wrong_gap1": score_prefix(
                compressed, model, conditions, fit_personal=personal
            ),
        },
        last,
        len(selected),
    )


def main():
    """Observed worker precedes terminal latent evaluation and oracle reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        model = json.loads((args.source / "source_model.json").read_text())
        rows = [
            json.loads(line)
            for line in (args.source / "application.jsonl").read_text().splitlines()
        ]
        outputs = []
        for author, events in grouped(rows).items():
            for budget in [4] if args.smoke else [16, 32, 48]:
                for personal in (False, True):
                    pair, last, count = paired_predictions(
                        events, model, budget, personal
                    )
                    for clock, result in pair.items():
                        outputs.append(
                            {
                                "author_id": author,
                                "budget": budget,
                                "last_t": last,
                                "observations": count,
                                "estimator": "personal_MAP"
                                if personal
                                else "source_only",
                                "clock": clock,
                                **result,
                            }
                        )
        dump(args.output / "observed_predictions.json", outputs)
        return
    if args.output.exists():
        raise FileExistsError("preserve previous runs")
    args.output.mkdir(parents=True)
    for item in json.loads((args.source / "output_manifest.json").read_text())["files"]:
        if sha256(args.source / item["path"]) != item["sha256"]:
            raise ValueError("source hash mismatch")
    summary = json.loads((args.source / "summary.json").read_text())
    if bool(summary["smoke"]) != args.smoke:
        raise ValueError("smoke/full mismatch")
    files = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f05_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f02_recovery.py",
    ]
    (args.output / "source_archive").mkdir()
    for path in files:
        (args.output / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        args.output / "input_manifest.json",
        {
            "source_manifest_sha256": sha256(args.source / "output_manifest.json"),
            "files": [{"path": str(p), "sha256": sha256(p)} for p in files],
        },
    )
    scores = []
    for outcome in summary["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        out = args.output / f"{world}_{seed}"
        out.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(root.resolve()),
            "--output",
            str(out.resolve()),
            "--worker",
        ]
        if args.smoke:
            command.append("--smoke")
        with (out / "worker.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        predictions = json.loads((out / "observed_predictions.json").read_text())
        targets = {
            r["author_id"]: r for r in json.loads((root / "truth.json").read_text())
        }
        people = grouped(
            [
                json.loads(line)
                for line in (root / "application.jsonl").read_text().splitlines()
            ]
        )
        for author, events in people.items():
            truth = targets[author]
            oracle = {
                "prior": {
                    "mu": [truth["a"], truth["b"], eta_from_rho(truth["rho"])],
                    "Sigma": np.eye(3).tolist(),
                },
                "log_sigma": np.log([truth["sigma_w"], truth["sigma_e"]]).tolist(),
            }
            for budget in [4] if args.smoke else [16, 32, 48]:
                pair, last, count = paired_predictions(events, oracle, budget, False)
                for clock, result in pair.items():
                    predictions.append(
                        {
                            "author_id": author,
                            "budget": budget,
                            "last_t": last,
                            "observations": count,
                            "estimator": "oracle",
                            "clock": clock,
                            **result,
                        }
                    )
        for row in predictions:
            truth = targets[row["author_id"]]
            last = row["last_t"]
            target = truth["states"][last]
            record = {
                k: row[k]
                for k in (
                    "author_id",
                    "budget",
                    "last_t",
                    "observations",
                    "estimator",
                    "clock",
                    "fit_success",
                )
            }
            record.update(
                world=world,
                seed=seed,
                state_mean=row["state_mean"],
                state_truth=target,
                state_MSE=(row["state_mean"] - target) ** 2,
                rho_estimate=row["rho"],
                rho_truth=truth["rho"],
                rho_MSE=(row["rho"] - truth["rho"]) ** 2,
                state_coverage=float(
                    abs(row["state_mean"] - target)
                    <= 1.959963984540054 * np.sqrt(row["state_variance"])
                ),
            )
            for k in (1, 4):
                prediction = row["forecasts"][k - 1]["mean"]
                future = people[row["author_id"]][last + k]
                conditional = (
                    truth["a"] + truth["b"] * future["z"] + truth["rho"] ** k * target
                )
                record.update(
                    {
                        f"forecast{k}_MSE": (prediction - future["y"]) ** 2,
                        f"conditional{k}_MSE": (prediction - conditional) ** 2,
                    }
                )
            scores.append(record)
        print(
            json.dumps({"world": world, "seed": seed, "rows": len(scores)}), flush=True
        )
    table(args.output / "missing_time_metrics.csv", scores)
    data = pd.DataFrame(scores)
    metrics = [
        "state_MSE",
        "rho_MSE",
        "forecast1_MSE",
        "forecast4_MSE",
        "conditional1_MSE",
        "conditional4_MSE",
        "state_coverage",
    ]
    data.groupby(["world", "budget", "estimator", "clock"])[
        metrics
    ].mean().reset_index().to_csv(args.output / "missing_time_summary.csv", index=False)
    contrasts = []
    for (world, budget, estimator), group in data.groupby(
        ["world", "budget", "estimator"]
    ):
        for metric in metrics[:6]:
            differences = {}
            for seed, subset in group.groupby("seed"):
                paired = subset.pivot(
                    index="author_id", columns="clock", values=metric
                ).sort_index()
                differences[seed] = (paired.correct_gap3 - paired.wrong_gap1).to_numpy()
            contrasts.append(
                {
                    "world": world,
                    "budget": int(budget),
                    "estimator": estimator,
                    "metric": metric,
                    **paired_bootstrap(differences),
                }
            )
    table(args.output / "paired_comparisons.csv", contrasts)
    dump(
        args.output / "summary.json",
        {
            "status": "MISSING_TIME_COMPARISON_EXECUTED_AUDIT_PENDING",
            "rows": len(scores),
            "contrasts": len(contrasts),
            "same_observations": True,
            "source_refitted": False,
            "time_anchor": "state at last actually observed t, forecast one/four original time units later",
            "interpretation": "wrong clock changes likelihood and interpretation; no universal benefit claim for finite-sample MAP",
            "remaining": "independent numerical replay and shared-episode uncertainty qualification",
        },
    )
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
