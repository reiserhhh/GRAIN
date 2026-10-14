#!/usr/bin/env python3
"""Independent likelihood, acquisition, outcome and metric replay for F06 A."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256


def jsonlines(path):
    """Stream compressed immutable records without modifying them."""
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def main():
    """Recompute all actions by enumerating both possible outcomes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier audit")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.input / "output_manifest.json").read_text())["files"]
    for item in manifest:
        assert sha256(args.input / item["path"]) == item["sha256"]
    corners = np.array(list(product((-1, 1), repeat=3)))
    laws = np.column_stack(
        [expit(b * corners[:, k]) for k in range(3) for b in (0.5, 1, 1.5)]
    )
    external = {r["author_id"]: r for r in jsonlines(args.input / "exogenous.jsonl.gz")}
    truth = {
        r["author_id"]: r for r in json.loads((args.input / "truth.json").read_text())
    }
    metrics = pd.read_csv(args.input / "policy_metrics.csv").set_index(
        ["author_id", "policy", "budget"]
    )
    maximum = 0.0
    checked_metrics = 0
    rows = 0
    key = None
    for row in jsonlines(args.input / "trajectories.jsonl.gz"):
        current = (row["author_id"], row["policy"])
        if current != key:
            key = current
            likelihood = np.ones(9)
            seen = set()
            previous_step = 0
        assert row["step"] == previous_step + 1
        previous_step = row["step"]
        w = likelihood / likelihood.sum()
        np.testing.assert_allclose(w, row["prior"], atol=1e-12, rtol=0)
        ext = external[row["author_id"]]
        target = truth[row["author_id"]]
        assert row["seed"] == ext["seed"] == target["seed"]
        assert row["role"] == ext["role"] == target["role"]
        policy = row["policy"]
        t = row["step"] - 1
        if policy == "fixed_corner":
            action = 0
        elif policy == "random":
            action = ext["random_actions"][t]
        elif policy == "coverage":
            action = t % 8
        else:
            probabilities = laws @ w
            if policy == "epistemic_variance":
                values = (laws**2) @ w - probabilities**2
            else:
                assert policy == "expected_entropy_gain"
                after_one = laws * w / probabilities[:, None]
                after_zero = (1 - laws) * w / (1 - probabilities[:, None])
                values = (
                    entropy(w)
                    - probabilities * entropy(after_one, axis=1)
                    - (1 - probabilities) * entropy(after_zero, axis=1)
                )
            action = int(
                np.flatnonzero(np.isclose(values, values.max(), atol=1e-14, rtol=0))[0]
            )
        assert action == row["action"], (key, t, action, row["action"])
        response = int(ext["uniforms"][t] < laws[action, target["atom"]])
        assert response == row["response"]
        likelihood *= laws[action] if response else 1 - laws[action]
        posterior = likelihood / likelihood.sum()
        maximum = max(maximum, float(np.max(np.abs(posterior - row["posterior"]))))
        np.testing.assert_allclose(posterior, row["posterior"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            entropy(posterior), row["entropy_after_update"], atol=1e-12, rtol=0
        )
        seen.add(action)
        index = (*key, row["step"])
        if index in metrics.index:
            actual = metrics.loc[index]
            prediction = laws @ posterior
            p = laws[:, target["atom"]]
            future_y = np.asarray(ext["future_uniforms"]) < p
            mse = np.mean((prediction - p) ** 2)
            expected = {
                "entropy": entropy(posterior),
                "atom_accuracy": float(
                    np.flatnonzero(
                        np.isclose(posterior, posterior.max(), atol=1e-14, rtol=0)
                    )[0]
                    == target["atom"]
                ),
                "family_accuracy": float(
                    np.flatnonzero(
                        np.isclose(
                            posterior.reshape(3, 3).sum(axis=1),
                            posterior.reshape(3, 3).sum(axis=1).max(),
                            atol=1e-14,
                            rtol=0,
                        )
                    )[0]
                    == target["axis"]
                ),
                "probability_MSE": mse,
                "expected_Brier": mse + np.mean(p * (1 - p)),
                "future_Brier": np.mean((prediction - future_y) ** 2),
                "distinct_corners": len(seen),
            }
            for name, value in expected.items():
                error = abs(value - actual[name])
                maximum = max(maximum, float(error))
                assert error < 1e-12, (index, name, value, actual[name])
            checked_metrics += 1
        rows += 1
    assert checked_metrics == len(metrics)
    selected = metrics.reset_index().query("role == 'application'")
    contrasts = pd.read_csv(args.input / "paired_comparisons.csv")
    for row in contrasts.itertuples(index=False):
        subset = selected[selected.budget == row.budget]
        rng = np.random.default_rng(20260915)
        means, samples = [], []
        for _, group in subset.groupby("seed", sort=True):
            paired = group.pivot(
                index="author_id", columns="policy", values=row.metric
            ).sort_index()
            delta = (paired[row.policy] - paired["random"]).to_numpy()
            means.append(delta.mean())
            samples.append(
                delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
            )
        low, high = np.quantile(np.mean(samples, axis=0), [0.025, 0.975])
        expected = np.array([np.mean(means), low, high])
        actual = np.array([row.delta, row.ci_low, row.ci_high])
        np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(expected - actual))))
    summary = {
        "status": "PANEL_A_ACTION_OUTCOME_POSTERIOR_METRIC_INTERVAL_REPLAY_PASS",
        "verified_hashes": len(manifest),
        "trajectory_rows": rows,
        "metric_rows": checked_metrics,
        "paired_intervals": len(contrasts),
        "maximum_numeric_difference": maximum,
        "scope": "Independent formulas in same-agent audit; no human or model mechanism validity claim. Other F06 panels remain open.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
