#!/usr/bin/env python3
"""Replay scalar CAT from count likelihoods and enumerate expected entropy."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record


def records(path):
    """Stream stored observations without changing their representation."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def first_max(values):
    """Independent fixed-order tie resolution."""
    return int(np.flatnonzero(np.abs(values - values.max()) <= 1e-14)[0])


def wilson(successes, count):
    """95% binomial interval for prior-predictive coverage, not human validity."""
    p, z = successes / count, 1.959963984540054
    center = (p + z**2 / (2 * count)) / (1 + z**2 / count)
    half = z * np.sqrt(p * (1 - p) / count + z**2 / (4 * count**2)) / (1 + z**2 / count)
    return center - half, center + half


def main():
    """Verify all actions, records, scores, intervals and finite-grid posteriors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior audits")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.input / "output_manifest.json").read_text())["files"]
    for item in manifest:
        assert sha256(args.input / item["path"]) == item["sha256"]
    protocol = json.loads((args.input / "protocol.json").read_text())
    grid = np.array(protocol["grid"])
    prior = np.array(protocol["prior"])
    difficulties = np.array(protocol["difficulties"])
    laws = expit(grid[None, :] - difficulties[:, None])
    log_prior = np.log(prior)
    splitmap = json.loads((args.input / "splitmap.json").read_text())
    truth = {
        r["author_id"]: r for r in json.loads((args.input / "truth.json").read_text())
    }
    external = {r["author_id"]: r for r in records(args.input / "exogenous.jsonl.gz")}
    budgets = {
        (r["author_id"], r["policy"], r["budget"]): np.array(r["weights"])
        for r in records(args.input / "budget_posteriors.jsonl.gz")
    }
    metrics = pd.read_csv(args.input / "metrics.csv").set_index(
        ["author_id", "policy", "budget"]
    )
    key = None
    seen = set()
    checked, metric_checks, rises = 0, 0, 0
    maximum = 0.0
    for row, common in zip(
        records(args.input / "trajectories.jsonl.gz"),
        records(args.input / "common_records.jsonl.gz"),
        strict=True,
    ):
        current = row["author_id"], row["policy"]
        if current != key:
            key = current
            successes, failures = np.zeros(3), np.zeros(3)
            weights = prior.copy()
            before_entropy = float(entropy(weights))
            next_t, visited = 0, set()
        assert row["t"] == next_t
        next_t += 1
        person = external[row["author_id"]]
        policy = row["policy"]
        if policy == "fixed_center":
            action = 1
        elif policy == "random":
            action = person["random_actions"][row["t"]]
        elif policy == "coverage":
            action = row["t"] % 3
        elif policy == "fisher_map":
            p = laws[:, first_max(weights)]
            action = first_max(p * (1 - p))
        else:
            assert policy == "expected_entropy_gain"
            p = laws @ weights
            after_yes = laws * weights / p[:, None]
            after_no = (1 - laws) * weights / (1 - p[:, None])
            score = (
                entropy(weights)
                - p * entropy(after_yes, axis=1)
                - (1 - p) * entropy(after_no, axis=1)
            )
            action = first_max(score)
        assert action == row["action"]
        atom = truth[row["author_id"]]["atom"]
        response = int(person["uniforms"][row["t"]] < laws[action, atom])
        assert response == row["response"]
        (successes if response else failures)[action] += 1
        log_weights = log_prior + successes @ np.log(laws) + failures @ np.log1p(-laws)
        weights = np.exp(log_weights - logsumexp(log_weights))
        mean = weights @ grid
        variance = weights @ grid**2 - mean**2
        ent = entropy(weights)
        low, high = grid[np.searchsorted(np.cumsum(weights), [0.025, 0.975])]
        expected_summary = {
            "mean": mean,
            "variance": variance,
            "entropy": ent,
            "ci_low": low,
            "ci_high": high,
            "boundary_mass": weights[0] + weights[-1],
        }
        for field, value in expected_summary.items():
            maximum = max(maximum, abs(value - row[field]))
            np.testing.assert_allclose(value, row[field], atol=1e-10, rtol=1e-8)
        if abs(ent - before_entropy) > 1e-12:
            assert bool(ent > before_entropy) == row["entropy_increase"]
        rises += int(row["entropy_increase"])
        before_entropy = ent
        validate_record(common, splitmap, seen)
        assert common["author_id"] == row["author_id"]
        assert common["role"] == row["role"] == person["role"]
        assert common["event_id"] == f"{row['author_id']}:{policy}:{row['t']}"
        assert common["t"] == row["t"] and common["observation"] == response
        assert common["condition"] == {"difficulty": difficulties[action]}
        visited.add(action)
        index = (*key, row["t"] + 1)
        if index in budgets:
            np.testing.assert_allclose(weights, budgets[index], atol=1e-12, rtol=0)
            p = laws[:, atom]
            prediction = laws @ weights
            probability_mse = np.mean((prediction - p) ** 2)
            expected = {
                **expected_summary,
                "a_MSE": (mean - grid[atom]) ** 2,
                "a_error": mean - grid[atom],
                "coverage": float(low <= grid[atom] <= high),
                "probability_MSE": probability_mse,
                "expected_Brier": probability_mse + np.mean(p * (1 - p)),
                "future_Brier": np.mean(
                    (prediction - (np.array(person["future_uniforms"]) < p)) ** 2
                ),
                "distinct_items": len(visited),
            }
            actual = metrics.loc[index]
            for field, value in expected.items():
                maximum = max(maximum, abs(value - actual[field]))
                np.testing.assert_allclose(value, actual[field], atol=1e-10, rtol=1e-8)
            metric_checks += 1
        checked += 1
    assert metric_checks == len(budgets) == len(metrics)
    selected = metrics.reset_index().query("role == 'application'")
    comparisons = pd.read_csv(args.input / "paired_comparisons.csv")
    for row in comparisons.itertuples(index=False):
        rng = np.random.default_rng(20260915)
        means, samples = [], []
        for _, group in selected[selected.budget == row.budget].groupby(
            "seed", sort=True
        ):
            pivot = group.pivot(
                index="author_id", columns="policy", values=row.metric
            ).sort_index()
            difference = (pivot[row.policy] - pivot[row.reference]).to_numpy()
            means.append(difference.mean())
            samples.append(
                difference[
                    rng.integers(0, len(difference), (2000, len(difference)))
                ].mean(axis=1)
            )
        low, high = np.quantile(np.mean(samples, axis=0), [0.025, 0.975])
        values = np.array([np.mean(means), low, high])
        stored = np.array([row.delta, row.ci_low, row.ci_high])
        np.testing.assert_allclose(values, stored, atol=1e-12, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(values - stored))))
    coverage = []
    for (budget, policy), group in selected.groupby(["budget", "policy"]):
        low, high = wilson(group.coverage.sum(), len(group))
        coverage.append(
            {
                "budget": budget,
                "policy": policy,
                "n": len(group),
                "coverage": group.coverage.mean(),
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    pd.DataFrame(coverage).to_csv(args.output / "coverage_intervals.csv", index=False)
    summary = {
        "status": "IRT_COUNT_LIKELIHOOD_AND_POLICY_REPLAY_PASS",
        "steps": checked,
        "metric_rows": metric_checks,
        "common_records": len(seen),
        "paired_intervals": len(comparisons),
        "verified_hashes": len(manifest),
        "maximum_numeric_difference": float(maximum),
        "entropy_increase_steps": rises,
        "scope": "same-agent independent formulas; discrete prior-predictive simulation, not empirical scale calibration",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
