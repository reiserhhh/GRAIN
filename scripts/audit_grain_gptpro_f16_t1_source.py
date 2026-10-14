#!/usr/bin/env python3
"""Re-fit four-cell source copies and audit all T1 application/reference mixtures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from grain_research.gptpro_foundation_v3.f04_recovery import fit_source_moments

CONDITIONS = (-1.0, -1 / 3, 1 / 3, 1.0)
CONTRAST = np.array([-1.0, 0.0, 0.0, 1.0])


def digest(path):
    """Hash inputs and score outputs in the post-score evidence record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped(path):
    """Read original synthetic records without changing their within-author order."""
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result.setdefault(row["author_id"], []).append(row)
    return result


def observation_posterior(rows, model):
    """Independent full observation-space solve, not the four-cell reduction."""
    schedule = [row["z"] for row in rows[0]]
    if any([row["z"] for row in person] != schedule for person in rows):
        raise ValueError("this audit batch requires matched observation schedules")
    x = np.eye(4)[[CONDITIONS.index(z) for z in schedule]]
    y = np.asarray([[row["y"] for row in person] for person in rows])
    mu, cov = np.asarray(model["mu"]), np.asarray(model["Sigma"])
    observation = x @ cov @ x.T + model["sigma2"] * np.eye(len(x))
    gain = np.linalg.solve(observation, x @ cov).T
    mean = mu + (y - x @ mu) @ gain.T
    posterior = cov - gain @ x @ cov
    return mean, (posterior + posterior.T) / 2


def replay_panel(rows, outputs, bundle, seed):
    """Audit every source-fit/author scalar and all prescribed joint conditional draws."""
    error = 0.0
    for b, model in enumerate(bundle["source_models"]):
        mean, covariance = observation_posterior(rows, model)
        predicted = mean @ CONTRAST
        variance = float(CONTRAST @ covariance @ CONTRAST)
        normals = np.random.default_rng(
            np.random.SeedSequence([seed, 16, 1, b])
        ).standard_normal((bundle["personal_draws"], 4))
        perturbations = normals @ np.linalg.cholesky(covariance).T
        projected = (mean[:, None, :] + perturbations[None, :, :]) @ CONTRAST
        mc_mean, mc_variance = projected.mean(axis=1), projected.var(axis=1, ddof=0)
        actual = np.asarray(
            [
                [
                    r["source_components"][b][key]
                    for key in (
                        "mean",
                        "conditional_variance",
                        "mc_mean",
                        "mc_variance",
                    )
                ]
                for r in outputs
            ]
        )
        expected = np.column_stack(
            [predicted, np.full(len(rows), variance), mc_mean, mc_variance]
        )
        error = max(error, float(np.max(np.abs(expected - actual))))
    for out in outputs:
        parts = out["source_components"]
        for prefix, mkey, vkey in (
            ("", "mean", "conditional_variance"),
            ("mc_", "mc_mean", "mc_variance"),
        ):
            means = np.asarray([p[mkey] for p in parts])
            within = float(np.mean([p[vkey] for p in parts]))
            between = float(np.mean(means * means) - means.mean() ** 2)
            error = max(
                error,
                abs(float(means.mean()) - out[prefix + "score"]),
                abs(within + between - out[prefix + "variance"]),
                abs(between - out[prefix + "components"]["source"]),
            )
    return error


def metrics(pred, variance, truth):
    """Report target-specific working intervals; finite coverage is not universal calibration."""
    radius = 1.96 * np.sqrt(variance)
    covered = np.abs(pred - truth) <= radius
    coverage = float(covered.mean())
    n, z = len(truth), float(norm.ppf(0.975))
    center = (coverage + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z
        * np.sqrt(coverage * (1 - coverage) / n + z * z / (4 * n * n))
        / (1 + z * z / n)
    )
    return {
        "mse": float(np.mean((pred - truth) ** 2)),
        "coverage": coverage,
        "mean_width": float(2 * radius.mean()),
        "wilson_95": [center - half, center + half],
        "decision": "PASS" if center - half <= 0.95 <= center + half else "FAIL",
    }


def main():
    """Audit original estimator equivalence before opening hidden evaluation targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    terminal = json.loads((args.run / "summary.json").read_text())
    if terminal["status"] != "T1_SOURCE_COLD_COMPLETE":
        raise ValueError("require completed T1 source run")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for cell in terminal["cells"]:
        seed = cell["seed"]
        folder = args.run / str(seed)
        root = args.source_root / f"L_{seed}"
        bundle = json.loads((folder / "bundle.json").read_text())
        source_fit = json.loads((folder / "source_fits.json").read_text())
        manifest = json.loads((folder / "application_manifest.json").read_text())
        if (
            digest(root / "source.jsonl") != bundle["source_hash"]
            or digest(root / "application.jsonl") != manifest["application_sha256"]
            or digest(folder / "bundle.json") != manifest["bundle_sha256"]
            or digest(folder / "source_fits.json") != bundle["source_fits_hash"]
            or digest(folder / "cold_scores.jsonl") != cell["scores_sha256"]
        ):
            raise ValueError("input, ensemble or scores changed")
        source = grouped(root / "source.jsonl")
        src_rows = [source[a] for a in source_fit["source_order"]]
        application = grouped(root / "application.jsonl")
        app_rows = [application[a] for a in manifest["author_order"]]
        source_error, floor_count = 0.0, 0
        for b, indices in enumerate(source_fit["indices"]):
            copies = [
                {**row, "author_id": f"copy{i}", "event_id": f"copy{i}/{j}"}
                for i, index in enumerate(indices)
                for j, row in enumerate(src_rows[index])
            ]
            independent = fit_source_moments(copies, "four_cell")
            saved = bundle["source_models"][b]
            for key in (
                "mu",
                "Sigma",
                "sigma2",
                "raw_eigenvalues",
                "sampling_covariance",
            ):
                source_error = max(
                    source_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(independent[key]) - np.asarray(saved[key])
                            )
                        )
                    ),
                )
            floor_count += int(min(saved["raw_eigenvalues"]) < 1e-6)
        scores = [
            json.loads(line)
            for line in (folder / "cold_scores.jsonl").read_text().splitlines()
        ]
        reference = json.loads((folder / "source_scores.json").read_text())
        if (
            len(scores) != cell["application_authors"]
            or len(reference) != cell["source_authors"]
        ):
            raise ValueError("incomplete target panel")
        reference_error = replay_panel(src_rows, reference, bundle, seed)
        application_error = replay_panel(app_rows, scores, bundle, seed)
        ref_values = np.asarray([s["score"] for s in reference])
        norm_error = max(
            abs(float(ref_values.mean()) - bundle["reference_mu"]),
            abs(float(ref_values.std(ddof=1)) - bundle["reference_sigma"]),
        )
        norm_error = max(
            norm_error,
            *(
                abs(
                    s["reference_score"]
                    - (s["score"] - bundle["reference_mu"]) / bundle["reference_sigma"]
                )
                for s in scores
            ),
            *(
                abs(
                    s["reference_variance"]
                    - s["variance"] / bundle["reference_sigma"] ** 2
                )
                for s in scores
            ),
        )
        if max(source_error, reference_error, application_error, norm_error) > 1e-10:
            raise AssertionError("T1 source-mixture numerical audit failed")
        # Evaluation-only hidden targets enter after score and fit verification.
        truth = {
            r["author_id"]: r
            for r in json.loads((root / "truth.json").read_text())
            if r["role"] == "application"
        }
        y = np.asarray(
            [
                truth[a]["means"][3] - truth[a]["means"][0]
                for a in manifest["author_order"]
            ]
        )
        analytic = np.asarray([[s["score"], s["variance"]] for s in scores])
        mc = np.asarray([[s["mc_score"], s["mc_variance"]] for s in scores])
        plugin = json.loads((folder / "plugin_model.json").read_text())
        pm, pv = observation_posterior(app_rows, plugin)
        result = {
            "seed": seed,
            "source_authors": len(src_rows),
            "application_authors": len(scores),
            "source_refits_replayed": len(source_fit["indices"]),
            "source_fits_with_spectral_floor": floor_count,
            "raw_min_eigenvalue": min(
                min(m["raw_eigenvalues"]) for m in bundle["source_models"]
            ),
            "max_source_refit_error": source_error,
            "max_reference_moment_error": reference_error,
            "max_application_moment_error": application_error,
            "max_normalization_error": norm_error,
            "conditional_draws_replayed": (len(src_rows) + len(scores))
            * bundle["source_bootstraps"]
            * bundle["personal_draws"],
            "analytic_source_mixture": metrics(analytic[:, 0], analytic[:, 1], y),
            "joint_conditional_mc_comparator": metrics(mc[:, 0], mc[:, 1], y),
            "plugin": metrics(
                pm @ CONTRAST, np.full(len(scores), CONTRAST @ pv @ CONTRAST), y
            ),
            "mean_source_variance": float(
                np.mean([s["components"]["source"] for s in scores])
            ),
            "mean_conditional_variance": float(
                np.mean([s["components"]["personal_conditional"] for s in scores])
            ),
            "mc_minus_analytic_score_rms": float(
                np.sqrt(np.mean((mc[:, 0] - analytic[:, 0]) ** 2))
            ),
            "cold_scores_sha256": cell["scores_sha256"],
            "truth_sha256": digest(root / "truth.json"),
        }
        results.append(result)
        (args.output / f"seed_{seed}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(result), flush=True)
    result = {
        "cells": results,
        "smoke": terminal["smoke"],
        "scope": "F04 L four-cell endpoint contrast; source bootstrap plus joint4D conditional200 comparator; no human validity",
        "parent_F16_complete": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
