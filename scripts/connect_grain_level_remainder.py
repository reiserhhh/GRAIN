#!/usr/bin/env python3
"""Join the new whole-score remainder to S9/S20 without fitting or neural calls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance


def main():
    """Preserve the exact configuration-plus-rest identity on identical authors."""
    out = BASE / "prefix_source_connection_v1"
    source = BASE / "source_remainder_s9_r1"
    if read(out / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete new source connection required")
    old = pd.read_csv(
        source / "private_components_before_outcomes.csv", dtype={"author_id": str}
    )
    new = pd.read_csv(out / "private_outcome_connection.csv", dtype={"author_id": str})
    frame = new.merge(
        old[
            [
                "author_id",
                "query",
                "new_teacher",
                "mean_score",
                "configuration_response",
            ]
        ],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(frame) != 277 or not np.allclose(
        frame.native, frame.new_teacher, rtol=0, atol=1e-12
    ):
        raise ValueError("S9/native teacher mismatch")
    frame["remainder"] = frame.native - frame.PQ
    frame["projected_minus_PQ"] = frame.mean_score - frame.PQ
    if not np.allclose(
        frame.remainder,
        frame.configuration_response + frame.projected_minus_PQ,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Configuration/remainder identity failed")
    source_pair = BASE / "functional_transfer_s20_r1/private_scores_before_outcomes.csv"
    pair = pd.read_csv(source_pair, dtype={"author_id": str})
    metrics = []
    for q, g in frame.groupby("query"):
        for col in (
            "PQ",
            "native",
            "remainder",
            "configuration_response",
            "projected_minus_PQ",
        ):
            metrics.append(
                {
                    "query": q,
                    "component": col,
                    "authors": len(g),
                    "r_with_remainder": correlation(
                        g[col].to_numpy(), g.remainder.to_numpy()
                    ),
                    "cov_y": covariance(g.y_true.to_numpy(), g[col].to_numpy()),
                    "r_y": correlation(g.y_true.to_numpy(), g[col].to_numpy()),
                }
            )
    n = frame[frame["query"] == "neuroticism"].merge(
        pair, on="author_id", validate="one_to_one"
    )
    if len(n) != 213:
        raise ValueError("S20 author mismatch")
    for name, g in (("all213", n), ("old_reference_excluded", n[~n.reference_author])):
        metrics.append(
            {
                "query": "neuroticism",
                "component": "S20_pairing_" + name,
                "authors": len(g),
                "r_with_remainder": correlation(
                    g.mean_response.to_numpy(), g.remainder.to_numpy()
                ),
                "cov_y": covariance(g.y_true.to_numpy(), g.mean_response.to_numpy()),
                "r_y": correlation(g.y_true.to_numpy(), g.mean_response.to_numpy()),
            }
        )
    frame.to_csv(out / "private_s9_remainder_connection.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "s9_remainder_connection.csv", index=False)
    write(
        out / "s9_remainder_summary.json",
        {
            "status": "COMPLETE",
            "neural_calls": 0,
            "fit_calls": 0,
            "new_labels": False,
            "s20_sha256": sha(source_pair),
            "metrics": metrics,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
