#!/usr/bin/env python3
"""Paired, post-freeze comparisons of the S9 source configuration contrast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_multiquery_sensitivity import corr
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, covariance


def main():
    """Use complete frozen scores only; no refits or selection from these intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(BASE):
        raise ValueError("Private outputs must remain ignored")
    summary = read(source / "summary.json")
    path = source / "private_components_before_outcomes.csv"
    if summary["status"] != "COMPLETE" or sha(path) != summary["components_sha256"]:
        raise ValueError("Complete frozen source components required")
    out = source / "paired_analysis"
    out.mkdir(exist_ok=False)
    frame = pd.read_csv(path, dtype={"author_id": str})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        usecols=["author_id", "query", "y_true"],
        dtype={"author_id": str},
    )
    frame = frame.merge(labels, on=["author_id", "query"], validate="one_to_one")
    pairs, nulls, accounting = [], [], []
    for query, g in frame.groupby("query"):
        for method, control in (
            ("new_teacher", "mean_score"),
            ("configuration_response", "new_teacher"),
            ("configuration_response", "student"),
        ):
            pairs.append(
                {
                    "query": query,
                    "method": method,
                    "control": control,
                    "metric": "auc" if query == "TF_cont" else "pearson_r",
                    **paired_intervals(
                        g.y_true, g[method], g[control], query == "TF_cont"
                    ),
                }
            )
        observed = corr(g.y_true, g.configuration_response)
        rng = np.random.default_rng(910731)
        draws = [
            corr(rng.permutation(g.y_true), g.configuration_response)
            for _ in range(2000)
        ]
        nulls.append(
            {
                "query": query,
                "observed_r": observed,
                "permutation_p_two_sided": float(
                    (1 + np.sum(np.abs(draws) >= abs(observed))) / 2001
                ),
                "null_mean_r": float(np.mean(draws)),
                "null_sd_r": float(np.std(draws)),
                "draws": len(draws),
            }
        )
        accounting.append(
            {
                "query": query,
                **{
                    name: covariance(g.y_true, g[name])
                    for name in (
                        "old_teacher",
                        "new_teacher",
                        "mean_score",
                        "student",
                        "observer_difference",
                        "configuration_response",
                        "mean_remainder",
                        "total_remainder",
                    )
                },
            }
        )
    for name, rows in (
        ("paired_intervals", pairs),
        ("permutation_control", nulls),
        ("covariance_totals", accounting),
    ):
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "script_sha256": sha(__file__),
            "source_components_sha256": sha(path),
            "new_fits": 0,
            "new_forwards": 0,
            "selection_on_outcomes": False,
            "scope": "pointwise exploratory paired intervals conditional on frozen source scores",
            "paired_intervals": pairs,
            "permutation_control": nulls,
            "covariance_totals": accounting,
        },
    )
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)
    print(pd.DataFrame(nulls).to_string(index=False), flush=True)
    print(pd.DataFrame(accounting).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
