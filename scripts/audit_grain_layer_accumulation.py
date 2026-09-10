#!/usr/bin/env python3
"""Replay NEXT42 saved files and independently sum each fixed early exit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_complete_transport_jvp import verify
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Verify byte replay, signed accounting and current-target-denied execution."""
    out = BASE / "logic_layer_accumulation_v1"
    verify(out)
    freeze = read(out / "prediction_freeze.json")
    verify_hashes(out, freeze["files"])
    summary = read(out / "analysis/summary.json")
    for name, digest in summary["files"].items():
        if any(
            sha(out / folder / name) != digest
            for folder in ("analysis", "analysis_replay")
        ):
            raise ValueError("Numerical CSV replay differs")
    error = 0.0
    for job in read(out / "private_jobs.json"):
        _, a = packet_arrays(out / "private_predictions" / (job["key"] + ".json"))
        if np.count_nonzero(a["initial"]):
            raise ValueError("Initial final-token tangent is not zero")
        for exit_index, update_count in enumerate((8, 14, 20, 21)):
            value = a["updates"][:, :update_count].astype(float).sum(1) + a["initial"]
            diff = value - a["endpoints"][:, exit_index]
            error = max(error, float(abs(diff).max()))
            relative = np.linalg.norm(diff) / max(np.linalg.norm(value), 1e-30)
            if not (relative <= 1e-4 or abs(diff).max() <= 1e-5):
                raise ValueError("An early exit differs from independent update sum")
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    comparison = pd.read_csv(out / "analysis/comparison.csv")
    if (
        len(actions) != 768
        or actions.duplicated(["key", "mode", "factor", "channel"]).any()
    ):
        raise ValueError("Action coverage differs")
    metric_error = 0.0
    for _, row in comparison.iterrows():
        a = actions
        for k in ("cohort", "query", "mode", "channel"):
            a = a[a[k] == row[k]]
        y, p = a.actual.to_numpy(), a.predicted.to_numpy()
        values = {
            "action_r": np.corrcoef(y, p)[0, 1],
            "action_mse_over_zero": np.square(y - p).sum() / np.square(y).sum(),
        }
        metric_error = max(metric_error, *(abs(values[k] - row[k]) for k in values))
    if metric_error > 1e-12:
        raise ValueError("Independent metric recalculation differs")
    stages = pd.read_csv(out / "analysis/stage_summary.csv")
    sums = stages.groupby(["cohort", "query", "mode"]).signed_projection_share.sum()
    share_error = float(abs(sums - 1).max())
    if share_error > 1e-5:
        raise ValueError("Signed projected updates do not sum to one")
    runtime = freeze["runtime"]
    if (
        runtime["current_scores"]
        or runtime["current_backwards"]
        or runtime["extra_denials"]
    ):
        raise ValueError("Target-denial contract differs")
    report = {
        "prediction_packets": 24,
        "exact_analysis_csvs": len(summary["files"]),
        "independent_exit_sum_max_absolute": error,
        "metric_replay_max": metric_error,
        "signed_projection_share_sum_error": share_error,
        "jvps": runtime["jvps"],
        "short_jvps": runtime["short_jvps"],
        "native_seconds": runtime["seconds"],
        "current_target_calls": 0,
    }
    write(out / "replay_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
