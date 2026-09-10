#!/usr/bin/env python3
"""Verify NEXT50 replay, statistical units and independently recomputed metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_query_selection import BASE, read, sha, verify, write


def audit(out):
    """Check only explicit frozen paths, then export aggregate public tables."""
    verify(out)
    primary = read(out / "predictions/freeze.json")
    replay = read(out / "predictions_replay/freeze.json")
    for record in (primary, replay):
        assert record["status"] == "COMPLETE"
        assert record["native_calls"] == record["fits"] == record["target_reads"] == 0
        assert record["current_Q_reads"] == record["extra_denials"] == 0
        assert record["denial_probes"] == 4 and not record["human_labels_read"]
    assert primary["files"] == replay["files"] and len(primary["files"]) == 12
    for name, expected in primary["files"].items():
        assert sha(out / "predictions" / name) == expected
        assert sha(out / "predictions_replay" / name) == expected
        with np.load(out / "predictions" / name, allow_pickle=False) as z:
            assert z["actions"].shape == (2, 3, 2, 4)
            np.testing.assert_array_equal(
                z["contrasts"], z["actions"][0] - z["actions"][1]
            )
    a = out / "analysis"
    r = out / "analysis_replay"
    summary = read(a / "summary.json")
    assert summary == read(r / "summary.json")
    for name, expected in summary["files"].items():
        assert sha(a / name) == sha(r / name) == expected
    df = pd.read_csv(a / "private_actions.csv")
    cf = pd.read_csv(a / "private_contrasts.csv")
    assert len(df) == 576 and len(cf) == 288
    counts = df.groupby("cohort").author_key.nunique().to_dict()
    assert counts == {"new8": 8, "source4": 4}
    public = (
        (df, ["cohort", "mode", "arm", "query_name"], "query_summary.csv"),
        (df, ["cohort", "mode", "arm", "query_name", "factor"], "factor_summary.csv"),
        (cf, ["cohort", "mode", "arm"], "contrast_summary.csv"),
        (cf, ["cohort", "mode", "arm", "factor"], "contrast_factors.csv"),
    )
    errors = []
    for frame, groups, filename in public:
        assert not frame.duplicated(
            ["author_key", "mode", "arm", "factor"]
            + (["query_name"] if "query_name" in frame else [])
        ).any()
        saved = pd.read_csv(a / filename).set_index(groups)
        for key, g in frame.groupby(groups, sort=True):
            actual, predicted = g.actual.to_numpy(), g.predicted.to_numpy()
            row = saved.loc[key]
            expected = {
                "action_r": np.corrcoef(actual, predicted)[0, 1],
                "action_mse_over_zero": np.sum((actual - predicted) ** 2)
                / np.sum(actual**2),
                "action_rmse": np.sqrt(np.mean((actual - predicted) ** 2)),
                "actual_action_rms": np.sqrt(np.mean(actual**2)),
                "predicted_action_rms": np.sqrt(np.mean(predicted**2)),
            }
            assert row.authors == g.author_key.nunique() and row.actions == len(g)
            assert row.same_sign == np.sum(np.sign(actual) == np.sign(predicted))
            errors.extend(abs(float(row[k]) - float(v)) for k, v in expected.items())
    assert max(errors) < 1e-10
    dst = ROOT / "reports/grain_vector_mechanisms/source_query_selection_tables"
    dst.mkdir(exist_ok=True)
    for _, _, name in public:
        (dst / name).write_bytes((a / name).read_bytes())
    result = {
        "status": "PASS",
        "array_replays": 12,
        "csv_replays": len(summary["files"]),
        "summary_replay": True,
        "independent_metric_max_error": max(errors),
        "author_counts": counts,
        "primary_seconds": primary["seconds"],
        "replay_seconds": replay["seconds"],
        "audit_code_sha256": sha(Path(__file__)),
        "public_tables": {name: sha(dst / name) for _, _, name in public},
    }
    write(out / "replay_audit.json", result)
    print(result)


if __name__ == "__main__":
    audit(BASE / "logic_source_query_selection_v1")
