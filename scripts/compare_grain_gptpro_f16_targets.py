#!/usr/bin/env python3
"""Paired author-level comparisons for frozen T1/T3 scores, without model selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f16_instrument import raw_t1


def paired_interval(values, seed=20260915, draws=2000):
    """Bootstrap authors within each seed; events are never independent samples."""
    rng = np.random.default_rng(seed)
    accumulated = np.zeros(draws)
    total = sum(len(array) for array in values)
    for array in values:
        array = np.asarray(array, dtype=float)
        sampled = rng.integers(0, len(array), size=(draws, len(array)))
        accumulated += array[sampled].sum(axis=1)
    samples = accumulated / total
    return {
        "mean_loss_difference": float(sum(np.sum(array) for array in values) / total),
        "paired_author_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "authors": total,
        "bootstrap_draws": draws,
        "scope": "seed-stratified paired authors; fixed predictions, not refitting",
    }


def hashed(path, expected):
    """Reject stale source/output bytes before pairing any author-level losses."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"evidence hash mismatch: {path}")


def main():
    """Retain oracle comparisons as references, never select a scorer using outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1-run", "t3-run", "t3-audit", "f04-source", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    differences = {
        "T1_analytic_minus_plugin": [],
        "T1_MC_minus_analytic": [],
        "T3_mixture_minus_known_parameter_oracle": [],
    }
    seed_rows = []
    for seed in (20260915, 20260916, 20260917):
        folder = args.t1_run / str(seed)
        manifest = json.loads((folder / "application_manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        bundle = json.loads((folder / "bundle.json").read_text())
        hashed(folder / "bundle.json", manifest["bundle_sha256"])
        hashed(folder / "cold_scores.jsonl", summary["scores_sha256"])
        source = args.f04_source / f"L_{seed}"
        hashed(source / "application.jsonl", manifest["application_sha256"])
        scores = [
            json.loads(line)
            for line in (folder / "cold_scores.jsonl").read_text().splitlines()
        ]
        people = {}
        for line in (source / "application.jsonl").read_text().splitlines():
            row = json.loads(line)
            people.setdefault(row["author_id"], []).append(row)
        truth = {
            row["author_id"]: row
            for row in json.loads((source / "truth.json").read_text())
            if row["role"] == "application"
        }
        # The mean source fit is not the plugin; use the original single fit.
        plugin = json.loads((folder / "plugin_model.json").read_text())
        ids = manifest["author_order"]
        y = np.asarray([truth[a]["means"][3] - truth[a]["means"][0] for a in ids])
        plug = np.asarray([raw_t1(people[a], plugin)[0] for a in ids])
        primary = np.asarray([row["score"] for row in scores])
        mc = np.asarray([row["mc_score"] for row in scores])
        if len(scores) != len(ids) or len(bundle["source_models"]) != 200:
            raise ValueError("incomplete T1 panel")
        current = {
            "T1_analytic_minus_plugin": (primary - y) ** 2 - (plug - y) ** 2,
            "T1_MC_minus_analytic": (mc - y) ** 2 - (primary - y) ** 2,
        }
        folder = args.t3_run / str(seed)
        summary = json.loads((folder / "summary.json").read_text())
        manifest = json.loads((folder / "application_manifest.json").read_text())
        hashed(folder / "cold_scores.jsonl", summary["cold_scores_sha256"])
        t3scores = [
            json.loads(line)
            for line in (folder / "cold_scores.jsonl").read_text().splitlines()
        ]
        outcomes = json.loads((args.t3_audit / f"outcomes_{seed}.json").read_text())
        if [row["author_id"] for row in outcomes] != manifest["author_order"]:
            raise ValueError("T3 outcome author alignment mismatch")
        y = np.asarray([row["planned_outcome"] for row in outcomes])
        oracle = np.asarray([row["oracle_conditional_mean"] for row in outcomes])
        primary = np.asarray([row["raw_score"] for row in t3scores])
        current["T3_mixture_minus_known_parameter_oracle"] = (primary - y) ** 2 - (
            oracle - y
        ) ** 2
        for key, array in current.items():
            differences[key].append(array)
            seed_rows.append(
                {"seed": seed, "comparison": key, **paired_interval([array])}
            )
    output = {
        "by_seed": seed_rows,
        "pooled": [
            {"comparison": key, **paired_interval(values)}
            for key, values in differences.items()
        ],
        "interpretation": "Loss differences only. CI crossing zero is not equivalence. No claim that a source-mixture estimator should beat a known-parameter oracle. No fitted model changed.",
        "parent_F16_complete": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(output["pooled"], indent=2))


if __name__ == "__main__":
    main()
