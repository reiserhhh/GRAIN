#!/usr/bin/env python3
"""Reconcile causal binding effects, collateral drift, and calibrated readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.interventions import fit_margin_threshold
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Do not treat an all-Yes threshold failure as proof of no retained information."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--span-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_status.json").exists():
        raise ValueError("Use a new output directory")
    status = json.loads((args.patch_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETE":
        raise ValueError("Patch not complete")
    predictions = pd.read_csv(args.parent_dir / "predictions.csv")
    calibrations = []
    for (order, task), sub in predictions.groupby(["order", "task"]):
        train = sub[sub.split.eq("discovery")]
        test = sub[sub.split.eq("heldout_scenario")]
        threshold = fit_margin_threshold(train.margin, train.expected_yes)
        calibrations.append(
            {
                "order": order,
                "task": task,
                "discovery_threshold": threshold,
                "heldout_default_accuracy": test.correct_default.mean(),
                "heldout_calibrated_balanced_accuracy": balanced_accuracy_score(
                    test.expected_yes, test.margin >= threshold
                ),
                "heldout_pooled_auc": roc_auc_score(test.expected_yes, test.margin),
                "heldout_scenarios": 4,
                "scope": "scoring calibration, not changed model reasoning",
            }
        )
    pd.DataFrame(calibrations).to_csv(out / "readout_calibration.csv", index=False)
    frame = pd.read_csv(args.patch_dir / "interventions.csv")
    summaries = []
    for (site, order, task), sub in frame.groupby(["site", "order", "task"]):
        metric = (
            "donor_truth_oriented_delta" if task == "narrator_positive" else "delta"
        )
        indexed = sub.pivot(index=["group_id", "case_id"], columns="arm", values=metric)
        random_columns = [k for k in indexed.columns if k.startswith("random_")]
        for arm in ("full", "rank2_half", "rank2_full"):
            values = indexed[arm]
            if task == "alias_action":
                values = abs(values)
            summaries.append(
                {
                    "site": site,
                    "order": order,
                    "task": task,
                    "contrast": arm,
                    "meaning": "donor-truth-oriented logit change"
                    if task == "narrator_positive"
                    else "absolute collateral logit change",
                    **grouped_bootstrap(
                        values.to_numpy(),
                        indexed.index.get_level_values("group_id").to_numpy(),
                    ),
                }
            )
        if task == "narrator_positive":
            values = indexed.rank2_full - indexed[random_columns].mean(axis=1)
            summaries.append(
                {
                    "site": site,
                    "order": order,
                    "task": task,
                    "contrast": "rank2_minus_norm_matched_random",
                    "meaning": "paired donor-truth-oriented logit change difference",
                    **grouped_bootstrap(
                        values.to_numpy(),
                        indexed.index.get_level_values("group_id").to_numpy(),
                    ),
                }
            )
    pd.DataFrame(summaries).to_csv(out / "patch_summary.csv", index=False)
    if args.span_dir:
        span_status = json.loads((args.span_dir / "run_status.json").read_text())
        if span_status["status"] != "COMPLETE":
            raise ValueError("Span patch not complete")
        span_frame = pd.read_csv(args.span_dir / "span_interventions.csv")
        span_summary = []
        for (site, order, task, span), sub in span_frame.groupby(
            ["site", "order", "task", "span"]
        ):
            metric = "oriented_delta" if task == "narrator_positive" else "delta"
            index = sub.pivot(
                index=["group_id", "case_id"], columns="arm", values=metric
            )
            raw = index["full"] if task == "narrator_positive" else abs(index["full"])
            scope = (
                "full_prompt_positive_control"
                if span == "all_prompt_control"
                else "selective_span_intervention"
            )
            span_summary.append(
                {
                    "site": site,
                    "order": order,
                    "task": task,
                    "span": span,
                    "contrast": "full",
                    "scope": scope,
                    **grouped_bootstrap(
                        raw.to_numpy(),
                        index.index.get_level_values("group_id").to_numpy(),
                    ),
                }
            )
            if task == "narrator_positive":
                delta = index["full"] - index[["random_0", "random_1"]].mean(axis=1)
                span_summary.append(
                    {
                        "site": site,
                        "order": order,
                        "task": task,
                        "span": span,
                        "contrast": "full_minus_token_norm_random",
                        "scope": scope,
                        **grouped_bootstrap(
                            delta.to_numpy(),
                            index.index.get_level_values("group_id").to_numpy(),
                        ),
                    }
                )
        pd.DataFrame(span_summary).to_csv(out / "span_summary.csv", index=False)
        print(pd.DataFrame(span_summary).to_string(index=False))
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "patch_run_status": status,
            "scope": "exploratory 4 scenario clusters; CIs conditional on discovered basis",
        },
    )
    print(pd.DataFrame(calibrations).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
