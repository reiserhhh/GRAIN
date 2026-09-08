#!/usr/bin/env python3
"""Recompute high-impact M22 comparisons and M22B accounting without new fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.nonlinear_readout import paired_mse_interval
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    paired_intervals,
    read,
    sha,
    write,
)


def main():
    """Audit saved predictions; cross-family contrasts do not select a new model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--edits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use ignored results for audit artifacts")
    out.mkdir(parents=True, exist_ok=False)
    source, edits = args.readout_dir, args.edits_dir
    path = source / "private_oof_before_outcomes.csv"
    if sha(path) != read(source / "oof_frozen.json")["sha256"]:
        raise ValueError("OOF hash mismatch")
    frame = pd.read_csv(path)
    units = ["cohort", "family", "query", "control", "author_id"]
    if frame.duplicated(units + ["method"]).any():
        raise ValueError("Duplicate author/method")
    if not (frame.groupby(units).fold.nunique() == 1).all():
        raise ValueError("Different reader folds for same author")
    if not (frame.groupby(units).teacher.nunique() == 1).all():
        raise ValueError("Different teacher values for same author")
    parts = frame[
        frame.method.isin(
            ["ridge", "stumps", "trees", "ridge_stumps", "ridge_trees", "inner_chosen"]
        )
    ]
    part_error = float(
        np.max(np.abs(parts.prediction - parts.linear_part - parts.nonlinear_part))
    )
    if part_error > 1e-12:
        raise ValueError("Reader additive ledger broken")
    linked = pd.read_csv(source / "private_outcome_connection.csv")
    ridge = linked[
        (linked.cohort == "prefix_matched")
        & (linked.control == "observed")
        & (linked.method == "ridge")
    ]
    paired = ridge[ridge.family == "prefix_pq_mean"].merge(
        ridge[ridge.family == "mean"],
        on=["author_id", "query"],
        validate="one_to_one",
        suffixes=("_prefix", "_e5"),
    )
    contrasts = []
    for query, group in paired.groupby("query"):
        if not np.array_equal(
            group.teacher_prefix, group.teacher_e5
        ) or not np.array_equal(group.fold_prefix, group.fold_e5):
            raise ValueError("Cross-family teacher or fold mismatch")
        contrasts.append(
            {
                "query": query,
                "authors": len(group),
                **paired_mse_interval(
                    group.teacher_prefix, group.prediction_prefix, group.prediction_e5
                ),
                **paired_intervals(
                    group.y_true_prefix,
                    group.prediction_prefix,
                    group.prediction_e5,
                    query == "TF_cont",
                ),
            }
        )
    pd.DataFrame(contrasts).to_csv(out / "representation_comparisons.csv", index=False)
    old = pd.read_csv(
        BASE / "student_learning_curve_s4_r1/private_oof_before_outcomes.csv",
        dtype={"size": str},
    )
    current = frame[
        (frame.cohort == "source_all")
        & (frame.control == "observed")
        & (frame.method == "ridge")
    ]
    joined = current.merge(
        old[old["size"] == "all"],
        on=["author_id", "query", "family"],
        validate="one_to_one",
        suffixes=("_current", "_old"),
    )
    if (
        len(joined) != 5082
        or not np.array_equal(joined.fold_current, joined.fold_old)
        or not np.array_equal(joined.teacher_current, joined.teacher_old)
    ):
        raise ValueError("S4 source or fold mismatch")
    replay = []
    for (query, family), g in joined.groupby(["query", "family"]):
        replay.append(
            {
                "query": query,
                "family": family,
                "authors": len(g),
                "max_prediction_difference": float(
                    np.max(np.abs(g.prediction_current - g.prediction_old))
                ),
                "current_r": float(g.teacher_current.corr(g.prediction_current)),
                "old_r": float(g.teacher_old.corr(g.prediction_old)),
            }
        )
    pd.DataFrame(replay).to_csv(out / "s4_replay_comparison.csv", index=False)
    cov = pd.read_csv(source / "covariance_account.csv")
    v = pd.read_csv(edits / "private_view_predictions.csv")
    interaction = v[v.condition == "interaction"]
    edit_error = float(
        np.max(
            np.abs(
                interaction.prediction
                - interaction.reader_interaction
                - interaction.feature_interaction
            )
        )
    )
    if edit_error > 1e-12 or cov.identity_error.max() > 1e-12:
        raise ValueError("Covariance or interaction identity broken")
    quality = {
        "status": "PASS_WITH_DISCOVERY_SCOPE",
        "new_fits": 0,
        "new_neural_calls": 0,
        "script_sha256": sha(__file__),
        "oof_sha256": sha(path),
        "unique_author_method_rows": len(frame),
        "matched_prefix_authors": len(paired),
        "s4_paired_rows": len(joined),
        "same_author_folds_and_teacher": True,
        "additive_reader_max_error": part_error,
        "covariance_max_error": float(cov.identity_error.max()),
        "natural_interaction_max_error": edit_error,
        "representation_comparisons": contrasts,
        "s4_replay": replay,
        "scope": "paired author bootstrap conditional on frozen fits; historical discovery labels; no causal neural-path claim",
    }
    write(out / "summary.json", quality)
    print(quality)


if __name__ == "__main__":
    main()
