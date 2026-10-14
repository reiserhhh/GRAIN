#!/usr/bin/env python3
"""Independent F06 B aggregation, paired intervals and descriptive Pareto ledger."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record


def paired_interval(data, arm, reference, metric):
    """Reconstruct the frozen seed-stratified author bootstrap independently."""
    rng = np.random.default_rng(20260915)
    means, draws, seed_rows = [], [], []
    for seed, group in data.groupby("seed", sort=True):
        pivot = group.pivot(
            index="author_id", columns="arm", values=metric
        ).sort_index()
        differences = (pivot[arm] - pivot[reference]).to_numpy()
        means.append(float(differences.mean()))
        draws.append(
            differences[
                rng.integers(0, len(differences), (2000, len(differences)))
            ].mean(axis=1)
        )
        seed_rows.append(
            {"seed": int(seed), "delta": means[-1], "authors": len(differences)}
        )
    low, high = np.quantile(np.mean(draws, axis=0), [0.025, 0.975])
    return {
        "delta": float(np.mean(means)),
        "ci_low": float(low),
        "ci_high": float(high),
    }, seed_rows


def pareto_rows(group):
    """Point-estimate nondominance only; no statistical superiority claim."""
    results = []
    for row in group.itertuples(index=False):
        dominators = group[
            (group.MSE <= row.MSE)
            & (group.sum_squared_perturbation <= row.sum_squared_perturbation)
            & (
                (group.MSE < row.MSE)
                | (group.sum_squared_perturbation < row.sum_squared_perturbation)
            )
        ]
        results.append(
            {
                "arm": row.arm,
                "MSE": row.MSE,
                "squared_perturbation": row.sum_squared_perturbation,
                "effective_actions": row.effective_actions,
                "point_nondominated": len(dominators) == 0,
                "dominated_by": "|".join(dominators.arm),
            }
        )
    return results


def main():
    """Require all role counts; partial tables cannot pass as full closure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads((args.input / "checkpoint.json").read_text())
    protocol = json.loads((args.input / "protocol.json").read_text())
    expected_authors = sum(protocol["counts"].values()) * len(protocol["seeds"])
    if (
        checkpoint["status"] != "COMPUTED_AUDIT_PENDING"
        or len(checkpoint["completed"]) != expected_authors
    ):
        raise ValueError(
            "full computation required; do not summarize partial role cohorts as complete"
        )
    if args.output.exists():
        raise FileExistsError("preserve prior evidence")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.input / "output_manifest.json").read_text())["files"]
    for item in manifest:
        assert sha256(args.input / item["path"]) == item["sha256"]
    metrics, fit_attempts = [], []
    seen, splits, counts = set(), {}, {}
    with gzip.open(args.output / "common_records.jsonl.gz", "wt") as common:
        for filename, digest in checkpoint["completed"].items():
            path = args.input / "authors" / filename
            assert sha256(path) == digest
            with gzip.open(path, "rt") as handle:
                packet = json.load(handle)
            metrics.extend(packet["metrics"])
            author, role = packet["author_id"], packet["role"]
            assert author not in splits
            splits[author] = role
            counts[(packet["seed"], role)] = counts.get((packet["seed"], role), 0) + 1
            for step in packet["steps"]:
                observation = step["observation"]
                episode = f"{author}:{step['reactivity']}:{step['arm']}"
                record = {
                    "author_id": author,
                    "event_id": f"{episode}:{step['t']}",
                    "episode_id": episode,
                    "t": step["t"],
                    "condition": {"z": observation["z"], "h": observation["h"]},
                    "action_timing": "after_observation_affects_next_state",
                    "opportunity": True,
                    "opportunity_status": "observed_then_next_action_selected",
                    "selected": True,
                    "observation": observation["y"],
                    "text_ref": "synthetic:no_text/gaussian_policy",
                    "reader_id": "f06_gaussian_scalar_observer",
                    "role": role,
                }
                validate_record(record, splits, seen)
                common.write(json.dumps(record) + "\n")
                fit_attempts.append(
                    {
                        "author_id": author,
                        "seed": packet["seed"],
                        "role": role,
                        "reactivity": step["reactivity"],
                        "arm": step["arm"],
                        "t": step["t"],
                        "success": step["fit_success"],
                        "attempts_converged": sum(
                            a["success"] for a in step["attempts"]
                        ),
                        "beta_status": step["beta_status"],
                    }
                )
    for seed in protocol["seeds"]:
        for role, expected in protocol["counts"].items():
            assert counts.get((seed, role), 0) == expected
    rebuilt = pd.DataFrame(metrics)
    stored = pd.read_csv(args.input / "metrics.csv")
    keys = ["author_id", "reactivity", "arm", "budget", "target"]
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    stored = stored.sort_values(keys).reset_index(drop=True)
    for field in rebuilt.columns:
        if pd.api.types.is_numeric_dtype(rebuilt[field]):
            np.testing.assert_allclose(
                rebuilt[field], stored[field], atol=1e-10, rtol=1e-8
            )
        else:
            assert (
                rebuilt[field].fillna("").tolist() == stored[field].fillna("").tolist()
            )
    application = rebuilt[rebuilt.role == "application"]
    group_keys = ["reactivity", "budget", "target", "arm"]
    measures = [
        "MSE",
        "bias",
        "variance",
        "coverage",
        "sum_squared_perturbation",
        "effective_actions",
        "fit_success",
        "prefix_fit_rate",
        "rho_MSE",
    ]
    for grouping, name in (
        (group_keys, "summary_by_policy.csv"),
        (["seed"] + group_keys, "summary_by_seed.csv"),
    ):
        rebuilt_summary = application.groupby(grouping)[measures].mean().reset_index()
        rebuilt_summary["RMSE"] = np.sqrt(rebuilt_summary.MSE)
        actual = pd.read_csv(args.input / name)
        pd.testing.assert_frame_equal(
            rebuilt_summary, actual, check_dtype=False, atol=1e-10, rtol=1e-8
        )
    summaries = application.groupby(group_keys)[measures].mean().reset_index()
    seed_differences, maximum = [], 0.0
    stored_intervals = pd.read_csv(args.input / "paired_comparisons.csv")
    for row in stored_intervals.itertuples(index=False):
        subset = application[
            (application.reactivity == row.reactivity)
            & (application.budget == row.budget)
            & (application.target == row.target)
        ]
        values, seed_rows = paired_interval(subset, row.arm, "random", row.metric)
        expected = np.array([values["delta"], values["ci_low"], values["ci_high"]])
        actual = np.array([row.delta, row.ci_low, row.ci_high])
        np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(expected - actual))))
        seed_differences.extend(
            {
                "reactivity": row.reactivity,
                "budget": row.budget,
                "target": row.target,
                "arm": row.arm,
                "metric": row.metric,
                **r,
            }
            for r in seed_rows
        )
    pd.DataFrame(seed_differences).to_csv(
        args.output / "seed_differences.csv", index=False
    )
    comparisons, pareto, coverage = [], [], []
    for (reactivity, budget, target), group in application.groupby(
        ["reactivity", "budget", "target"]
    ):
        for reference in ("fixed", "coverage", "random"):
            arm = "scalar_" + target
            for metric in ("MSE", "sum_squared_perturbation"):
                values, _ = paired_interval(group, arm, reference, metric)
                comparisons.append(
                    {
                        "reactivity": reactivity,
                        "budget": budget,
                        "target": target,
                        "arm": arm,
                        "reference": reference,
                        "metric": metric,
                        **values,
                        "classification": "LOWER"
                        if values["ci_high"] < 0
                        else "HIGHER"
                        if values["ci_low"] > 0
                        else "UNRESOLVED",
                    }
                )
        selected = summaries[
            (summaries.reactivity == reactivity)
            & (summaries.budget == budget)
            & (summaries.target == target)
        ]
        pareto.extend(
            {"reactivity": reactivity, "budget": budget, "target": target, **r}
            for r in pareto_rows(selected)
        )
        for arm, observations in group.groupby("arm"):
            n, p, z = len(observations), observations.coverage.mean(), 1.959963984540054
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            coverage.append(
                {
                    "reactivity": reactivity,
                    "budget": budget,
                    "target": target,
                    "arm": arm,
                    "coverage": p,
                    "wilson_low": center - width,
                    "wilson_high": center + width,
                    "n": n,
                }
            )
    pd.DataFrame(comparisons).to_csv(
        args.output / "target_specific_comparisons.csv", index=False
    )
    pd.DataFrame(pareto).to_csv(args.output / "descriptive_pareto.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "coverage_intervals.csv", index=False)
    fits = pd.DataFrame(fit_attempts)
    fits.groupby(["seed", "role", "reactivity", "arm"])[
        ["success", "attempts_converged"]
    ].mean().reset_index().to_csv(args.output / "fit_rates.csv", index=False)
    result = {
        "status": "FULL_AGGREGATION_INTERVAL_COMMON_RECORD_REPLAY_PASS",
        "authors": len(splits),
        "common_records": len(seen),
        "metrics": len(rebuilt),
        "original_intervals": len(stored_intervals),
        "target_specific_contrasts": len(comparisons),
        "max_interval_replay_difference": maximum,
        "failed_prefixes": int((~fits.success).sum()),
        "verified_hashes": len(manifest),
        "scope": "descriptive point Pareto; approximate posterior coverage empirical; scientific verdict must consider each target separately",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
