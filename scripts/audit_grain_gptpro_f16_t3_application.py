#!/usr/bin/env python3
"""Audit frozen T3 scores, then evaluate the declared planned future outcome."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from grain_research.gptpro_foundation_v3 import f16_independent_t3 as t3
from grain_research.gptpro_foundation_v3.f16_t3_application import (
    dense_future,
    moment_discrepancy,
    planned_outcomes,
)


def digest(path):
    """Bind post-score evaluation to the actual frozen prediction bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_metrics(pred, lo, hi, outcome):
    """Separate empirical predictive coverage from numerical implementation checks."""
    n = len(outcome)
    covered = (lo <= outcome) & (outcome <= hi)
    p = float(covered.mean())
    z = float(norm.ppf(0.975))
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "mse": float(np.mean((pred - outcome) ** 2)),
        "coverage": p,
        "mean_width": float((hi - lo).mean()),
        "wilson_95": [center - half, center + half],
        "decision": "PASS" if center - half <= 0.95 <= center + half else "FAIL",
    }


def main():
    """All score audits precede opening latent truth for each seed cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "build-root", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    terminal = json.loads((args.run / "summary.json").read_text())
    if terminal["status"] != "T3_COLD_REPLAY_COMPLETE":
        raise ValueError("completed cold run required")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for cell in terminal["cells"]:
        seed = cell["seed"]
        root = args.run / str(seed)
        build = args.build_root / f"core_{seed}"
        protocol = json.loads((root / "protocol_before_application.json").read_text())
        bundle = t3.load_bundle(
            root / "bundle.json", protocol["bundle_sha256"], protocol["source_hash"]
        )
        if digest(build / "bundle.json") != protocol["bundle_sha256"]:
            raise ValueError("independent source bundle changed")
        manifest = json.loads((root / "application_manifest.json").read_text())
        source = args.source_root / f"core_{seed}"
        if digest(source / "application.jsonl") != manifest["sha256"]:
            raise ValueError("application source changed")
        if digest(root / "cold_scores.jsonl") != cell["cold_scores_sha256"]:
            raise ValueError("scored output changed")
        outputs = [
            json.loads(line)
            for line in (root / "cold_scores.jsonl").read_text().splitlines()
        ]
        ids = manifest["author_order"]
        if len(outputs) != cell["application_authors"] or len(ids) != len(outputs):
            raise ValueError("incomplete application")
        checkpoint = json.loads((build / "checkpoint.json").read_text())
        reference = checkpoint["records"]
        if (
            t3.object_hash(reference) != bundle["reference_records_hash"]
            or len(reference) != 300
        ):
            raise ValueError("source reference changed or incomplete")
        all_scores = [record["score"] for record in reference] + outputs
        moment_error = max(moment_discrepancy(score, bundle) for score in all_scores)
        values = np.asarray([record["score"]["raw_score"] for record in reference])
        norm_error = max(
            abs(float(values.mean()) - bundle["reference_mu"]),
            abs(float(values.std(ddof=1)) - bundle["reference_sigma"]),
        )
        groups = {}
        for line in (source / "application.jsonl").read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row["author_id"], []).append(row)
        for i, author in enumerate(ids):
            raw = {
                "world": "core",
                "reader_hash": bundle["reader_hash"],
                "prefix": groups[author][:48],
            }
            if t3.object_hash(raw) != manifest["input_hashes"][i]:
                raise ValueError("observed prefix changed")
        dense_error = 0.0
        dense_count = 0
        for i in sorted({0, len(ids) - 1}):
            rows = groups[ids[i]][:48]
            observations = t3.prepare_input(
                {"world": "core", "reader_hash": bundle["reader_hash"], "prefix": rows},
                bundle,
            )
            for b in (0, 100, 199):
                model = bundle["ensemble"][b]
                params, _ = t3.personal_draws(observations, model, seed, 200)
                if params is None:
                    raise ValueError("replayed parameter update unsupported")
                means, variances = np.asarray(
                    [
                        dense_future(psi, rows, model["fit"]["log_sigma"])
                        for psi in params
                    ]
                ).T
                saved = outputs[i]["per_source"][b]
                dense_error = max(
                    dense_error,
                    abs(float(means.mean()) - saved["mean"]),
                    abs(float(variances.mean()) - saved["event_state"]),
                    abs(
                        float(np.mean(means * means) - means.mean() ** 2)
                        - saved["personal_parameter"]
                    ),
                )
                dense_count += len(params)
        if max(moment_error, norm_error, dense_error) > 1e-10:
            raise AssertionError("T3 numerical audit failed")
        # Latent parameters are opened ONLY after all predictions and numeric checks.
        truth = {
            row["author_id"]: row
            for row in json.loads((source / "truth.json").read_text())
            if row["role"] == "application"
        }
        outcomes, historical, oracle_mean, oracle_var = [], [], [], []
        outcome_records = []
        for author in ids:
            actual = truth[author]
            rows = groups[author]
            future = rows[48:52]
            if [row["t"] for row in future] != [48, 49, 50, 51]:
                raise ValueError("future must be four consecutive steps")
            yplan = planned_outcomes(future, actual["b"], bundle["future_conditions"])
            yold = np.asarray([row["y"] for row in future])
            state = np.asarray(actual["states"][48:52])
            errors = (
                yold
                - actual["a"]
                - actual["b"] * np.asarray([row["z"] for row in future])
                - state
            )
            reconstructed = (
                actual["a"]
                + actual["b"] * np.asarray(bundle["future_conditions"])
                + state
                + errors
            )
            if not np.allclose(yplan, reconstructed, rtol=0, atol=1e-12):
                raise AssertionError("planned structural outcome identity failed")
            outcomes.append(float(yplan.mean()))
            historical.append(float(yold.mean()))
            om, ov = dense_future(
                [actual["a"], actual["b"], np.arctanh(actual["rho"] / 0.99)],
                rows[:48],
                np.log([actual["sigma_w"], actual["sigma_e"]]),
            )
            oracle_mean.append(om)
            oracle_var.append(ov)
            outcome_records.append(
                {
                    "author_id": author,
                    "planned_outcome": outcomes[-1],
                    "historical_outcome": historical[-1],
                    "historical_conditions": [row["z"] for row in future],
                    "oracle_conditional_mean": om,
                    "oracle_predictive_variance": ov,
                }
            )
        pred = np.asarray([score["raw_score"] for score in outputs])
        lo, hi = np.asarray([score["interval"] for score in outputs]).T
        oracle_mean, oracle_var = np.asarray(oracle_mean), np.asarray(oracle_var)
        outcomes = np.asarray(outcomes)
        historical = np.asarray(historical)
        result = {
            "seed": seed,
            "application_authors": len(ids),
            "target": "planned_four_step_observation_mean",
            "future_conditions": bundle["future_conditions"],
            "planned_outcome": interval_metrics(pred, lo, hi, outcomes),
            "mismatched_historical_outcome_diagnostic_only": interval_metrics(
                pred, lo, hi, historical
            ),
            "known_parameter_oracle": interval_metrics(
                oracle_mean,
                oracle_mean - 1.96 * np.sqrt(oracle_var),
                oracle_mean + 1.96 * np.sqrt(oracle_var),
                outcomes,
            ),
            "prediction_vs_oracle_conditional_mean_mse": float(
                np.mean((pred - oracle_mean) ** 2)
            ),
            "planned_minus_historical_mean": float((outcomes - historical).mean()),
            "planned_minus_historical_rms": float(
                np.sqrt(np.mean((outcomes - historical) ** 2))
            ),
            "max_moment_error": moment_error,
            "max_reference_error": norm_error,
            "dense_parameter_draws": dense_count,
            "max_dense_error": dense_error,
            "cold_scores_sha256": digest(root / "cold_scores.jsonl"),
            "truth_sha256": digest(source / "truth.json"),
        }
        (args.output / f"outcomes_{seed}.json").write_text(
            json.dumps(outcome_records, sort_keys=True, allow_nan=False) + "\n"
        )
        summaries.append(result)
    result = {
        "cells": summaries,
        "smoke": terminal["smoke"],
        "scope": "SIM F05 core planned intervention; additive condition does not change state/noise; predictive interval, not conditional-mean CI",
        "numeric_scope": "all stored mixture moments and references; selected dense joint checks; not independent MAP optimization",
        "parent_F16_complete": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
