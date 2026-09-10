#!/usr/bin/env python3
"""Aggregate NEXT19/20 fidelity first, then post-freeze old source-criterion links."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.joint_readout import vector_quality
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.query_functional import response_quality
from scripts.run_grain_joint_carrier_readout import check, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def interval(values):
    """Pointwise finite-panel percentile interval, filtering degenerate resamples."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return list(np.quantile(a, [0.025, 0.975])) if len(a) else [float("nan")] * 2


def indices(authors):
    """Resample complete authors, keeping adjacent/distant pairs attached."""
    authors = np.asarray(authors)
    unique = np.unique(authors)
    groups = [np.flatnonzero(authors == a) for a in unique]
    rng = np.random.default_rng(20260910)
    return [
        np.concatenate([groups[i] for i in draw])
        for draw in rng.integers(0, len(unique), (1999, len(unique)))
    ]


def joint_metrics(records):
    """Retain both fixed feature families and simple baselines, no winning-arm selection."""
    rows = []
    for arm in sorted({r["arm"] for r in records}):
        for pair in ("all", "adjacent", "distant"):
            g = [
                r
                for r in records
                if r["arm"] == arm and (pair == "all" or r["pair"] == pair)
            ]
            g = sorted(g, key=lambda r: (r["author_id"], r["pair"]))
            a, p = [
                np.array([r[k] for r in g]) for k in ("actual_mixed", "predicted_mixed")
            ]
            draws = indices([r["author_id"] for r in g])
            q = vector_quality(a, p)
            errors = [np.sum((a[ix] - p[ix]) ** 2) / np.sum(a[ix] ** 2) for ix in draws]
            norms = [
                correlation(
                    np.linalg.norm(a[ix], axis=1), np.linalg.norm(p[ix], axis=1)
                )
                for ix in draws
            ]
            mse_ci, norm_ci = interval(errors), interval(norms)
            actual = np.array([r["actual_score"] for r in g])
            pred_score = np.array([r["oracle_singletons_reconstruction"] for r in g])
            zero_lookup = {
                (r["author_id"], r["pair"]): r["oracle_singletons_reconstruction"]
                for r in records
                if r["arm"] == "zero"
            }
            zero_score = np.array([zero_lookup[r["author_id"], r["pair"]] for r in g])
            score_energy = np.sum((actual - zero_score) ** 2)
            rows.append(
                {
                    "arm": arm,
                    "pair": pair,
                    "n_authors": len({r["author_id"] for r in g}),
                    "n_pairs": len(g),
                    **q,
                    "mse_over_zero_low": mse_ci[0],
                    "mse_over_zero_high": mse_ci[1],
                    "norm_r_low": norm_ci[0],
                    "norm_r_high": norm_ci[1],
                    "oracle_singletons_score_rmse": float(
                        np.sqrt(np.mean((actual - pred_score) ** 2))
                    ),
                    "oracle_singletons_score_error_over_additive": float(
                        np.sum((actual - pred_score) ** 2) / score_energy
                    ),
                }
            )
    return pd.DataFrame(rows)


def carrier_metrics(data):
    """Evaluate reference transfer primarily on the20 authors supplying no gradients."""
    rows = []
    for scope in ("evaluation", "all"):
        selected = data[data.role == "evaluation"] if scope == "evaluation" else data
        for arm, g in selected.groupby("arm"):
            g = g.sort_values("author_id")
            a, p = g.actual_retained.to_numpy(), g.predicted_retained.to_numpy()
            draws = indices(g.author_id)
            values = [np.sum((a[ix] - p[ix]) ** 2) / np.sum(a[ix] ** 2) for ix in draws]
            correlations = [correlation(a[ix], p[ix]) for ix in draws]
            ci, rci = interval(values), interval(correlations)
            rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    **response_quality(a, p),
                    "r": correlation(a, p),
                    "r_low": rci[0],
                    "r_high": rci[1],
                    "error_energy_low": ci[0],
                    "error_energy_high": ci[1],
                    "oracle_baseline_body_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    g.actual_complete_b
                                    - g.oracle_baseline_body_reconstruction
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def criterion_links(joint, carrier, outcomes):
    """Connect pre-frozen source-author readouts; questionnaire Y never trains maps."""
    rows = []
    lookup = {(r.author_id, r.query): r.y_true for r in outcomes.itertuples()}
    for arm in sorted({r["arm"] for r in joint}):
        for pair in ("adjacent", "distant"):
            g = sorted(
                [r for r in joint if r["arm"] == arm and r["pair"] == pair],
                key=lambda r: r["author_id"],
            )
            y = np.array([lookup[r["author_id"], r["query"]] for r in g])
            for component in (
                "predicted_mixed_norm",
                "actual_mixed_norm",
                "oracle_singletons_reconstruction",
            ):
                if component.endswith("_norm"):
                    x = np.linalg.norm(
                        np.array([r[component.removesuffix("_norm")] for r in g]),
                        axis=1,
                    )
                else:
                    x = np.array([r[component] for r in g])
                ci = interval(
                    [
                        correlation(x[ix], y[ix])
                        for ix in indices([r["author_id"] for r in g])
                    ]
                )
                rows.append(
                    {
                        "query": "neuroticism",
                        "scope": pair,
                        "arm": arm,
                        "component": component,
                        "n": len(y),
                        "r_y": correlation(x, y),
                        "low": ci[0],
                        "high": ci[1],
                    }
                )
    for arm, g in carrier[carrier.role == "evaluation"].groupby("arm"):
        g = g.sort_values("author_id")
        y = np.array([lookup[r.author_id, r.query] for r in g.itertuples()])
        for component in (
            "actual_retained",
            "predicted_retained",
            "actual_complete_b",
            "oracle_baseline_body_reconstruction",
        ):
            x = g[component].to_numpy()
            ci = interval([correlation(x[ix], y[ix]) for ix in indices(g.author_id)])
            rows.append(
                {
                    "query": "TF_cont",
                    "scope": "evaluation",
                    "arm": arm,
                    "component": component,
                    "n": len(y),
                    "r_y": correlation(x, y),
                    "low": ci[0],
                    "high": ci[1],
                }
            )
    return pd.DataFrame(rows)


def analyze(out, name):
    """Verify freezes, aggregate model fidelity, then open permitted source outcomes."""
    manifest = verify(out)
    for filename in ("joint_freeze.json", "carrier_freeze.json"):
        frozen = read(out / filename)
        if frozen["human_outcomes_read"] is not False:
            raise ValueError("Expected pre-criterion numerical freeze")
        for p, digest in frozen["files"].items():
            check(out / p, digest)
    dest = out / name
    dest.mkdir(exist_ok=False)
    joint = read(out / "private_joint_predictions.json")
    carrier = pd.read_csv(
        out / "private_carrier_predictions.csv", dtype={"author_id": str}
    )
    jmetrics, cmetrics = joint_metrics(joint), carrier_metrics(carrier)
    jmetrics.to_csv(dest / "joint_metrics.csv", index=False)
    cmetrics.to_csv(dest / "carrier_metrics.csv", index=False)
    path = BASE / "source_anchor_diagnosis_v1/private_anchor_source_connection.csv"
    check(path, manifest["source_outcome_sha256"])
    outcomes = pd.read_csv(
        path, usecols=["author_id", "query", "y_true"], dtype={"author_id": str}
    )
    if outcomes.duplicated(["author_id", "query"]).any():
        raise ValueError("Unique old author/query outcomes required")
    links = criterion_links(joint, carrier, outcomes)
    links.to_csv(dest / "source_criterion_links.csv", index=False)
    audits = read(out / "private_integration_audit.json")
    summary = {
        "status": "COMPLETE",
        "human_outcomes_used_for_fitting": False,
        "old_N_authors": 24,
        "TF_reference_authors": 4,
        "TF_evaluation_authors": 20,
        "cost": read(out / "carrier_status.json"),
        "integration_paths": len(audits),
        "paths_refined": sum(r["nodes"] == 5 for r in audits),
        "path_tolerance_passes": sum(r["passes"] for r in audits),
        "endpoint_replay_max": max(r["endpoint_error"] for r in audits),
        "integration_error_max": max(r["final_error"] for r in audits),
        "matrix": read(out / "carrier_matrix_summary.json"),
        "question": "Reference/author-disjoint transfer of conditional model actions, not a new personality benchmark",
        "files": {
            p.name: sha(p)
            for p in (
                dest / "joint_metrics.csv",
                dest / "carrier_metrics.csv",
                dest / "source_criterion_links.csv",
            )
        },
    }
    write(dest / "summary.json", summary)
    print(summary)
    print(
        jmetrics[jmetrics.pair == "all"][
            [
                "arm",
                "mse_over_zero",
                "norm_r",
                "oracle_singletons_score_error_over_additive",
            ]
        ].to_string(index=False)
    )
    print(
        cmetrics[cmetrics.scope == "evaluation"][
            ["arm", "error_energy_over_zero", "r", "same_sign"]
        ].to_string(index=False)
    )


def main():
    """Keep all generated author-linked data below the ignored research directory."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_joint_carrier_readout_v1"
    )
    p.add_argument("--name", default="analysis_r1")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Ignored output and simple analysis name required")
    analyze(out, args.name)


if __name__ == "__main__":
    main()
