#!/usr/bin/env python3
"""Audit completed T2 construction and evaluate truth only after scores freeze."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def digest(path):
    """Hash archived inputs, including the already frozen cold output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_state(parameters, rows, logs):
    """Independently condition the full stationary observation covariance."""
    a, b, eta = parameters
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.array(logs))
    times = np.array([row["t"] for row in rows])
    residual = np.array([row["y"] - a - b * row["z"] for row in rows])
    stationary = q / (1 - rho * rho)
    latent = stationary * rho ** np.abs(times[:, None] - times[None, :])
    covariance = latent + r * np.eye(len(rows))
    cross = latent[-1]
    solution = np.linalg.solve(covariance, np.column_stack([residual, cross]))
    return float(cross @ solution[:, 0]), float(stationary - cross @ solution[:, 1])


def raw_moment_check(result):
    """Recompute all mixture parts with raw second moments, not centered sums."""
    parts = result["source_components"]
    center = sum(part["mean"] for part in parts) / len(parts)
    source = sum(part["mean"] ** 2 for part in parts) / len(parts) - center**2
    event = sum(part["event_state"] for part in parts) / len(parts)
    personal = sum(part["personal_parameter"] for part in parts) / len(parts)
    expected = {"source": source, "event_state": event, "personal_parameter": personal}
    return max(
        abs(result["score"] - center),
        abs(result["variance"] - sum(expected.values())),
        *(abs(expected[k] - result["components"][k]) for k in expected),
    )


def main():
    """Keep arithmetic audit distinct from new interval calibration evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    terminal = json.loads((args.run / "summary.json").read_text())
    if terminal["status"] != "T2_COLD_REPLAY_COMPLETE":
        raise ValueError("require completed run")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for cell in terminal["cells"]:
        seed = cell["seed"]
        folder = args.run / str(seed)
        bundle = json.loads((folder / "bundle.json").read_text())
        application = json.loads((folder / "application_manifest.json").read_text())
        if digest(folder / "bundle.json") != application["bundle_sha256"]:
            raise ValueError("bundle changed after scoring")
        source = root = args.source_root / f"core_{seed}"
        if digest(source / "application.jsonl") != application["input_hash"]:
            raise ValueError("application input changed")
        if digest(source / "source.jsonl") != bundle["source_hash"]["source"]:
            raise ValueError("source reference changed")
        outputs = [
            json.loads(line)
            for line in (folder / "cold_scores.jsonl").read_text().splitlines()
        ]
        reference = [
            json.loads(path.read_text())["result"]
            for path in sorted((folder / "source_scores").glob("[0-9]*.json"))
        ]
        if (
            len(outputs) != cell["application_authors"]
            or len(reference) != cell["source_authors"]
        ):
            raise ValueError("count mismatch")
        max_moment = max(raw_moment_check(result) for result in reference + outputs)
        ref = np.array([result["score"] for result in reference])
        norm_error = max(
            abs(float(ref.mean()) - bundle["reference_mu"]),
            abs(
                float(np.sqrt(np.sum((ref - ref.mean()) ** 2) / (len(ref) - 1)))
                - bundle["reference_sigma"]
            ),
        )
        normalized_error = max(
            max(
                abs(
                    (r["score"] - bundle["reference_mu"]) / bundle["reference_sigma"]
                    - r["reference_score"]
                ),
                abs(
                    r["reference_variance"]
                    - r["variance"] / bundle["reference_sigma"] ** 2
                ),
            )
            for r in outputs
        )
        groups = {}
        for line in (root / "application.jsonl").read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row["author_id"], []).append(row)
        # Predetermined representative covariance checks; not all MAP/Hessian refits.
        dense_error = 0.0
        dense_count = 0
        indices = sorted({0, len(outputs) - 1})
        bs = sorted(
            {0, len(bundle["source_models"]) // 2, len(bundle["source_models"]) - 1}
        )
        for i in indices:
            rows = groups[application["author_order"][i]][:48]
            for b in bs:
                saved = outputs[i]["source_components"][b]
                model = bundle["source_models"][b]["model"]
                recipe = [seed, 16, 2, b]
                normals = np.random.default_rng(np.random.SeedSequence(recipe)).normal(
                    size=(bundle["personal_draws"], 3)
                )
                params = (
                    np.array(saved["parameter_center"])
                    + normals @ np.linalg.cholesky(saved["parameter_covariance"]).T
                )
                m, v = np.array(
                    [dense_state(par, rows, model["log_sigma"]) for par in params]
                ).T
                dense_error = max(
                    dense_error,
                    abs(float(m.mean()) - saved["mean"]),
                    abs(float(v.mean()) - saved["event_state"]),
                    abs(
                        float(np.mean(m * m) - m.mean() ** 2)
                        - saved["personal_parameter"]
                    ),
                )
                dense_count += len(params)
        if max(max_moment, norm_error, normalized_error, dense_error) > 1e-10:
            raise AssertionError("construction numerical audit failed")
        # Truth only enters this separate post-scoring evaluator.
        truth = {
            r["author_id"]: r
            for r in json.loads((root / "truth.json").read_text())
            if r["role"] == "application"
        }
        ys = np.array([truth[a]["states"][47] for a in application["author_order"]])
        pred = np.array([r["score"] for r in outputs])
        lo = np.array([r["lower"] for r in outputs])
        hi = np.array([r["upper"] for r in outputs])
        n = len(ys)
        hits = int(np.sum((lo <= ys) & (ys <= hi)))
        coverage = hits / n
        z = float(norm.ppf(0.975))
        denom = 1 + z * z / n
        center = (coverage + z * z / (2 * n)) / denom
        half = z * np.sqrt(coverage * (1 - coverage) / n + z * z / (4 * n * n)) / denom
        summaries.append(
            {
                "seed": seed,
                "n_authors": n,
                "target": "current_state",
                "prefix": 48,
                "scope": "existing synthetic core application; new numerical integration rule",
                "cold_scores_sha256": digest(folder / "cold_scores.jsonl"),
                "truth_sha256": digest(root / "truth.json"),
                "mean_reference": bundle["reference_mu"],
                "sd_reference": bundle["reference_sigma"],
                "mse": float(np.mean((pred - ys) ** 2)),
                "coverage": coverage,
                "mean_width": float(np.mean(hi - lo)),
                "wilson_95": [center - half, center + half],
                "decision": "PASS"
                if center - half <= 0.95 <= center + half
                else "FAIL",
                "max_moment_error": max_moment,
                "max_reference_error": max(norm_error, normalized_error),
                "dense_selected_draws": dense_count,
                "max_dense_error": dense_error,
            }
        )
    result = {
        "cells": summaries,
        "smoke": terminal["smoke"],
        "audit_scope": "all saved mixture moments and source references; selected dense filters; no MAP-refit audit",
        "calibration_scope": "post-score finite core state checks, not transferred to arbitrary input",
        "full_F16_acceptance": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
