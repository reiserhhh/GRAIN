#!/usr/bin/env python3
"""F13 joint dynamic target check on existing F05 data; no new parameter fitting."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import grouped
from grain_research.gptpro_foundation_v3.f05_recovery import dense_posterior
from grain_research.gptpro_foundation_v3.f13_dynamic import (
    joint_forecast,
    target_projection,
)
from scripts.audit_grain_gptpro_f13_nested import wilson


def write_table(path, rows):
    """Write nonempty scalar records."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Reuse source-fit/prefix parameters, simulate full correlated future vectors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    worlds = ["core"] if args.smoke else ["core", "near_root", "shared_shock"]
    budgets = [16] if args.smoke else [16, 32, 48]
    draws = 10 if args.smoke else 200
    config = {
        "seeds": seeds,
        "worlds": worlds,
        "budgets": budgets,
        "draws": draws,
        "targets": [
            "current_state",
            "future_observation_k4",
            "future_observation_average4",
        ],
        "scope": "oracle and frozen plug-in conditional event law; source/parameter bootstrap pending",
        "shared_shock": "marginal individual filter, not joint-population oracle; author Wilson invalid under cross-author dependence",
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    manifest, metrics, predictions = [], [], []
    max_state_error = max_forecast_error = max_dense_error = 0.0
    simulations = 0
    projection = target_projection()
    for wi, world in enumerate(worlds):
        for seed in seeds:
            directory = args.source / f"{world}_{seed}"
            cached_path = (
                args.application / f"{world}_{seed}" / "observed_predictions.json"
            )
            for path in (
                directory / "application.jsonl",
                directory / "source_model.json",
                directory / "truth.json",
                cached_path,
            ):
                manifest.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            model = json.loads((directory / "source_model.json").read_text())
            people = grouped(
                [
                    json.loads(line)
                    for line in (directory / "application.jsonl")
                    .read_text()
                    .splitlines()
                ]
            )
            cached = {
                (r["author_id"], r["n"]): r
                for r in json.loads(cached_path.read_text())
                if r["method"] == "personal_MAP" and r["phase"] == "prefix"
            }
            # Truth only supplies oracle comparison and final outcome evaluation.
            truth = {
                r["author_id"]: r
                for r in json.loads((directory / "truth.json").read_text())
            }
            ids = sorted(people)[:6] if args.smoke else sorted(people)
            local = []
            for ni, n in enumerate(budgets):
                for i, author in enumerate(ids):
                    rows = people[author]
                    zfuture = [r["z"] for r in rows[n : n + 4]]
                    actual = truth[author]
                    observed_target = np.array(
                        [
                            actual["states"][n - 1],
                            rows[n + 3]["y"],
                            np.mean([r["y"] for r in rows[n : n + 4]]),
                        ]
                    )
                    plugin = cached[(author, n)]
                    for mi, method in enumerate(("plugin", "oracle")):
                        psi = np.array(
                            plugin["psi"]
                            if method == "plugin"
                            else [
                                actual["a"],
                                actual["b"],
                                np.arctanh(actual["rho"] / 0.99),
                            ]
                        )
                        logs = np.array(
                            model["log_sigma"]
                            if method == "plugin"
                            else np.log([actual["sigma_w"], actual["sigma_e"]])
                        )
                        state = dense_posterior(psi, rows[:n], logs)
                        mean, cov, loading, variances = joint_forecast(
                            psi,
                            logs,
                            state["state_mean"],
                            state["state_variance"],
                            zfuture,
                        )
                        if method == "plugin":
                            max_state_error = max(
                                max_state_error,
                                abs(state["state_mean"] - plugin["state_mean"]),
                                abs(state["state_variance"] - plugin["state_variance"]),
                            )
                            for k, old in enumerate(plugin["forecasts"][:4], start=1):
                                max_forecast_error = max(
                                    max_forecast_error,
                                    abs(mean[k] - old["mean"]),
                                    abs(cov[k, k] - old["variance"]),
                                )
                        # Alternate unconditional AR block followed by Schur complement.
                        rho = 0.99 * np.tanh(psi[2])
                        q, r = np.exp(2 * logs)
                        times = np.array([row["t"] for row in rows[:n]])
                        target_times = times[-1] + np.arange(5)
                        stationary = q / (1 - rho * rho)
                        oo = stationary * rho ** np.abs(
                            times[:, None] - times
                        ) + r * np.eye(n)
                        to = stationary * rho ** np.abs(target_times[:, None] - times)
                        tt = stationary * rho ** np.abs(
                            target_times[:, None] - target_times
                        ) + np.diag([0, r, r, r, r])
                        max_dense_error = max(
                            max_dense_error,
                            float(
                                np.max(
                                    np.abs(cov - (tt - to @ np.linalg.solve(oo, to.T)))
                                )
                            ),
                        )
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, 13, 5, wi, ni, i, mi])
                        )
                        samples = (
                            mean
                            + (
                                rng.normal(size=(draws, len(variances)))
                                * np.sqrt(variances)
                            )
                            @ loading.T
                        )
                        target_samples = samples @ projection.T
                        simulations += draws
                        exact_mean, exact_cov = (
                            projection @ mean,
                            projection @ cov @ projection.T,
                        )
                        mc_mean, mc_var = (
                            target_samples.mean(axis=0),
                            target_samples.var(axis=0),
                        )
                        for ti, target in enumerate(config["targets"]):
                            for arm, m, v in (
                                ("joint_analytic", exact_mean[ti], exact_cov[ti, ti]),
                                ("joint_mc", mc_mean[ti], mc_var[ti]),
                            ):
                                local.append(
                                    {
                                        "world": world,
                                        "seed": seed,
                                        "budget": n,
                                        "author_id": author,
                                        "method": method,
                                        "arm": arm,
                                        "target": target,
                                        "mean": m,
                                        "variance": v,
                                        "truth": observed_target[ti],
                                        "fit_success": bool(plugin["fit_success"])
                                        if method == "plugin"
                                        else True,
                                    }
                                )
                        naive = float(np.sum(np.diag(cov)[1:]) / 16)
                        local.append(
                            {
                                "world": world,
                                "seed": seed,
                                "budget": n,
                                "author_id": author,
                                "method": method,
                                "arm": "incorrect_diagonal_only",
                                "target": config["targets"][2],
                                "mean": exact_mean[2],
                                "variance": naive,
                                "truth": observed_target[2],
                                "fit_success": bool(plugin["fit_success"])
                                if method == "plugin"
                                else True,
                            }
                        )
            write_table(args.output / f"predictions_{world}_{seed}.csv", local)
            predictions.extend(local)
            (args.output / "checkpoint.json").write_text(
                json.dumps(
                    {"world": world, "seed": seed, "simulation_vectors": simulations}
                )
                + "\n"
            )
    groups = {}
    for row in predictions:
        key = tuple(
            row[k] for k in ("world", "seed", "budget", "method", "arm", "target")
        )
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        mean, var, true = (
            np.array([r[name] for r in rows]) for name in ("mean", "variance", "truth")
        )
        ok = np.abs(mean - true) <= 1.96 * np.sqrt(var)
        low, high = wilson(int(ok.sum()), len(rows))
        dependent = key[0] == "shared_shock"
        metrics.append(
            dict(
                zip(
                    ("world", "seed", "budget", "method", "arm", "target"),
                    key,
                    strict=True,
                )
            )
            | {
                "n": len(rows),
                "mse": float(np.mean((mean - true) ** 2)),
                "coverage": float(ok.mean()),
                "mean_width": float(np.mean(3.92 * np.sqrt(var))),
                "wilson_low": low,
                "wilson_high": high,
                "calibration_decision": "DEPENDENT_AUTHORS_DESCRIPTIVE_ONLY"
                if dependent
                else ("PASS" if low <= 0.95 <= high else "FAIL"),
                "fit_failures": sum(not r["fit_success"] for r in rows),
            }
        )
    write_table(args.output / "metrics.csv", metrics)
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    summary = {
        "status": "DYNAMIC_JOINT_CONDITIONAL_TARGETS_COMPUTED_OUTER_BOOTSTRAP_OPEN",
        "rows": len(predictions),
        "simulation_vectors": simulations,
        "max_cached_state_error": max_state_error,
        "max_cached_forecast_error": max_forecast_error,
        "max_dense_covariance_error": max_dense_error,
        "source_refits": 0,
        "remaining": "source200/person200 hierarchical mixture; missingness/history/shift; full draw and table replay",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if max(max_state_error, max_forecast_error, max_dense_error) > 1e-8:
        raise ValueError("cached/dense disagreement; retain artifacts")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
