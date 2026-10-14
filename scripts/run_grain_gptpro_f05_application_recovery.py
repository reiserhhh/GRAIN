#!/usr/bin/env python3
"""F05 frozen-source application forecasts with isolated observed-only workers."""

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
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_lmm,
    score_response,
)
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_recovery import score_prefix
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def read_rows(path):
    """Load a single declared observed-data file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def observed_worker(source, out, smoke=False):
    """No truth file or future outcome is passed to any estimation function."""
    model = json.loads((source / "source_model.json").read_text())
    people = grouped(read_rows(source / "source.jsonl"))
    source_rows = [r for events in people.values() for r in events[:48]]
    lmm = fit_source_lmm(source_rows)
    dump(out / "source_lmm.json", lmm)
    forecasts = []
    for author, events in grouped(read_rows(source / "application.jsonl")).items():
        prefixes = [4] if smoke else [16, 32, 48]
        frozen = None
        for n in prefixes + ([] if smoke else list(range(49, 64))):
            prefix = events[:n]
            # z is fixed by the design clock, not obtained from future outcomes.
            conditions = [
                -1.0 if ((n + k) // 4) % 2 == 0 else 1.0
                for k in range(min(4, len(events) - n))
            ]
            if n in prefixes:
                personal = score_prefix(prefix, model, conditions)
                static = np.linalg.lstsq(
                    np.array([[1, r["z"]] for r in prefix]),
                    np.array([r["y"] for r in prefix]),
                    rcond=None,
                )[0]
                hierarchy = score_response(prefix, lmm)["coefficients"]
                frozen = (personal, static, hierarchy)
            else:
                prior_personal, static, hierarchy = frozen
                frozen_model = {
                    "log_sigma": model["log_sigma"],
                    "prior": {
                        "mu": prior_personal["psi"],
                        "Sigma": model["prior"]["Sigma"],
                    },
                }
                personal = score_prefix(
                    prefix, frozen_model, conditions, fit_personal=False
                )
                personal["fit_success"] = prior_personal["fit_success"]
            population = score_prefix(prefix, model, conditions, fit_personal=False)
            methods = {"personal_MAP": personal, "source_only_dynamics": population}
            for name, beta in (("static_OLS", static), ("source_LMM", hierarchy)):
                methods[name] = {
                    "psi": [float(beta[0]), float(beta[1]), None],
                    "rho": None,
                    "fit_success": not lmm["fallback"]
                    if name == "source_LMM"
                    else True,
                    "state_mean": 0.0,
                    "state_variance": None,
                    "forecasts": [
                        {
                            "k": k + 1,
                            "mean": float(beta[0] + beta[1] * z),
                            "variance": None,
                        }
                        for k, z in enumerate(conditions)
                    ],
                }
            for method, result in methods.items():
                forecasts.append(
                    {
                        "author_id": author,
                        "n": n,
                        "phase": "prefix" if n in prefixes else "sequential",
                        "method": method,
                        **result,
                    }
                )
    dump(out / "observed_predictions.json", forecasts)


def main():
    """Fit only the missing source LMM; freeze and score with terminal truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        observed_worker(args.source, args.output, args.smoke)
        return
    if args.output.exists():
        raise FileExistsError("preserve previous output")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.source / "output_manifest.json").read_text())
    for entry in manifest["files"]:
        if sha256(args.source / entry["path"]) != entry["sha256"]:
            raise ValueError("source manifest mismatch")
    summary = json.loads((args.source / "summary.json").read_text())
    if bool(summary["smoke"]) != args.smoke:
        raise ValueError("smoke/full mismatch")
    files = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f05_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f04_recovery.py",
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
    scores, forecasts = [], []
    for outcome in summary["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        source = args.source / f"{world}_{seed}"
        out = args.output / f"{world}_{seed}"
        out.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(source.resolve()),
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
        # Only after the observed worker has terminated do latent targets enter.
        targets = {
            r["author_id"]: r for r in json.loads((source / "truth.json").read_text())
        }
        people = grouped(read_rows(source / "application.jsonl"))
        predictions = json.loads((out / "observed_predictions.json").read_text())
        for author, events in people.items():
            target = targets[author]
            oracle_model = {
                "prior": {
                    "mu": [target["a"], target["b"], eta_from_rho(target["rho"])],
                    "Sigma": np.eye(3).tolist(),
                },
                "log_sigma": np.log([target["sigma_w"], target["sigma_e"]]).tolist(),
            }
            for n in [4] if args.smoke else [16, 32, 48, *range(49, 64)]:
                conditions = [
                    -1.0 if ((n + k) // 4) % 2 == 0 else 1.0
                    for k in range(min(4, len(events) - n))
                ]
                result = score_prefix(
                    events[:n], oracle_model, conditions, fit_personal=False
                )
                predictions.append(
                    {
                        "author_id": author,
                        "n": n,
                        "phase": "prefix" if n in (4, 16, 32, 48) else "sequential",
                        "method": "oracle_Kalman",
                        **result,
                    }
                )
        for row in predictions:
            target = targets[row["author_id"]]
            n = row["n"]
            state = target["states"][n - 1]
            metadata = {
                key: row[key]
                for key in ("author_id", "n", "phase", "method", "fit_success")
            }
            metadata.update(world=world, seed=seed)
            variance = row["state_variance"]
            scores.append(
                {
                    **metadata,
                    "a_squared_error": (row["psi"][0] - target["a"]) ** 2,
                    "b_squared_error": (row["psi"][1] - target["b"]) ** 2,
                    "rho_squared_error": (row["rho"] - target["rho"]) ** 2
                    if row["rho"] is not None
                    else None,
                    "state_squared_error": (row["state_mean"] - state) ** 2,
                    "state_mean": row["state_mean"],
                    "state_truth": state,
                    "state_variance": variance,
                    "state_covered": float(
                        abs(row["state_mean"] - state)
                        <= 1.959963984540054 * np.sqrt(variance)
                    )
                    if variance is not None
                    else None,
                }
            )
            for prediction in row["forecasts"]:
                k = prediction["k"]
                if k not in (1, 4):
                    continue
                future = people[row["author_id"]][n + k - 1]
                conditional = (
                    target["a"] + target["b"] * future["z"] + target["rho"] ** k * state
                )
                forecasts.append(
                    {
                        **metadata,
                        "k": k,
                        "prediction": prediction["mean"],
                        "future_y": future["y"],
                        "conditional_truth": conditional,
                        "forecast_squared_error": (prediction["mean"] - future["y"])
                        ** 2,
                        "conditional_squared_error": (prediction["mean"] - conditional)
                        ** 2,
                    }
                )
        dump(
            args.output / "checkpoint.json",
            {
                "last_world": world,
                "last_seed": seed,
                "state_rows": len(scores),
                "forecast_rows": len(forecasts),
            },
        )
        print(
            json.dumps({"world": world, "seed": seed, "state_rows": len(scores)}),
            flush=True,
        )
    table(args.output / "state_parameter_metrics.csv", scores)
    table(args.output / "forecast_metrics.csv", forecasts)
    data = pd.DataFrame(scores)
    metrics = [
        "a_squared_error",
        "b_squared_error",
        "rho_squared_error",
        "state_squared_error",
        "state_covered",
    ]
    data.groupby(["world", "n", "phase", "method"])[
        metrics
    ].mean().reset_index().to_csv(args.output / "state_summary.csv", index=False)
    pd.DataFrame(forecasts).groupby(["world", "n", "phase", "method", "k"])[
        ["forecast_squared_error", "conditional_squared_error"]
    ].mean().reset_index().to_csv(args.output / "forecast_summary.csv", index=False)
    contrasts = []
    for world in sorted(data.world.unique()):
        for metric in metrics[:4]:
            selected = data[
                (data.world == world) & (data.n == (4 if args.smoke else 48))
            ]
            differences = {}
            for seed, group in selected.groupby("seed"):
                pairs = group.pivot(
                    index="author_id", columns="method", values=metric
                ).sort_index()
                differences[seed] = (
                    pairs.personal_MAP - pairs.source_only_dynamics
                ).to_numpy()
            contrasts.append(
                {"world": world, "metric": metric, **paired_bootstrap(differences)}
            )
    table(args.output / "paired_comparisons.csv", contrasts)
    dump(
        args.output / "summary.json",
        {
            "status": "APPLICATION_COMPUTED_AUDIT_PENDING",
            "smoke": args.smoke,
            "state_rows": len(scores),
            "forecast_rows": len(forecasts),
            "source_dynamics_refitted": False,
            "source_LMM_newly_fit": True,
            "sequential_rule": "freeze personal parameters at48, update state with preceding observations only",
            "remaining": [
                "joint shared covariance channel",
                "missing-time experiment",
                "source-fixed profiles Hessians",
                "independent metric and leakage audit",
                "task packaging",
            ],
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
