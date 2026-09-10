#!/usr/bin/env python3
"""Independently audit NEXT54 frozen cells, raw pairings and signed allocations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_response_repair import (
    ARMS,
    BASE,
    CELLS,
    OBJECTIVES,
    OLD,
    QUERIES,
    arrays,
    frozen,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)


def audit(out):
    """Use raw differences, not the analysis helper, to reconstruct every account."""
    m = verify(out)
    pf, rf = frozen(out, "predictions"), frozen(out, "replay_predictions")
    jobs = read(out / "private_jobs.json")
    packets, diagonal_error = 0, 0.0
    for j in jobs:
        for key in j["query_keys"]:
            a = arrays(out / "predictions" / (key + ".npz"))
            b = arrays(out / "replay_predictions" / (key + ".npz"))
            if set(a) != {"vectors", "actions"} or set(a) != set(b):
                raise ValueError("Exact two-field prediction packet required")
            for field in a:
                np.testing.assert_array_equal(a[field], b[field])
            for si, objective in enumerate(OBJECTIVES):
                old = arrays(OLD / objective / "predictions" / (key + ".npz"))
                for ai in range(2):
                    for field in a:
                        diff = a[field][ai, si, si] - old[field][ai + 1]
                        diagonal_error = max(diagonal_error, float(abs(diff).max()))
            packets += 1
    if packets != 24 or diagonal_error > 1e-11:
        raise ValueError("Frozen diagonal inventory or formula differs")
    summaries = [
        read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for folder, summary in zip(("analysis", "replay_analysis"), summaries, strict=True):
        verify_hashes(out / folder, summary["files"])
    if set(summaries[0]["files"]) != set(summaries[1]["files"]):
        raise ValueError("Replay table inventory differs")
    for name in summaries[0]["files"]:
        if sha(out / "analysis" / name) != sha(out / "replay_analysis" / name):
            raise ValueError("Table replay differs: " + name)

    df = pd.read_csv(out / "analysis/private_cells.csv")
    cf = pd.read_csv(out / "analysis/private_contrast_cells.csv")
    idx = ["author_key", "cohort", "family", "cell", "factor"]
    expected = (
        df[df.query_name == QUERIES[0]].set_index(idx)[["actual", "predicted"]]
        - df[df.query_name == QUERIES[1]].set_index(idx)[["actual", "predicted"]]
    ).sort_index()
    np.testing.assert_allclose(
        expected, cf.set_index(idx)[["actual", "predicted"]].sort_index(), atol=1e-14
    )
    metric_error, accounting_error = 0.0, 0.0
    for tab, groups, effect_file, table_files in (
        (
            df,
            ["cohort", "family", "query_name"],
            "private_effects.csv",
            ("query", "factor"),
        ),
        (
            cf,
            ["cohort", "family"],
            "private_contrast_effects.csv",
            ("contrast", "contrast_factor"),
        ),
    ):
        unit = ["author_key", *groups, "factor"]
        stored = pd.read_csv(out / "analysis" / effect_file).set_index(unit)
        for key, g in tab.groupby(unit, sort=True):
            vv, vj, jv, jj = g.set_index("cell").loc[list(CELLS), "predicted"]
            actual = g.actual.iloc[0]
            e = vv - actual
            s, t = (jv - vv + jj - vj) / 2, (vj - vv + jj - jv) / 2
            check = {
                "actual": actual,
                "value": vv,
                "joint": jj,
                "state": s,
                "tangent": t,
                "interaction": jj - jv - vj + vv,
                "change": jj - vv,
                "base_squared_error": e * e,
                "joint_squared_error": (jj - actual) ** 2,
                "state_linear_error": 2 * e * s,
                "tangent_linear_error": 2 * e * t,
                "state_squared": s * s,
                "tangent_squared": t * t,
                "cross_error": 2 * s * t,
                "error_change": (jj - actual) ** 2 - e * e,
            }
            accounting_error = max(
                accounting_error,
                max(abs(float(stored.loc[key, k]) - v) for k, v in check.items()),
            )
        for extra, stem in zip(([], ["factor"]), table_files, strict=True):
            grouping = groups + extra
            metrics = pd.read_csv(out / "analysis" / (stem + "_cells.csv")).set_index(
                grouping + ["cell"]
            )
            for key, g in tab.groupby(grouping + ["cell"], sort=True):
                y, p = g.actual.to_numpy(), g.predicted.to_numpy()
                mse = np.mean((p - y) ** 2)
                check = {
                    "action_r": np.corrcoef(y, p)[0, 1],
                    "action_mse_over_zero": mse / np.mean(y * y),
                    "action_rmse": np.sqrt(mse),
                    "actual_action_rms": np.sqrt(np.mean(y * y)),
                    "predicted_action_rms": np.sqrt(np.mean(p * p)),
                }
                metric_error = max(
                    metric_error,
                    max(abs(float(metrics.loc[key, k]) - v) for k, v in check.items()),
                )
            effects = pd.read_csv(out / "analysis" / (stem + "_effects.csv")).set_index(
                grouping
            )
            for key, g in stored.reset_index().groupby(grouping, sort=True):
                y, vv, jj = g.actual.to_numpy(), g.value.to_numpy(), g.joint.to_numpy()
                e, delta = vv - y, jj - vv
                s, t = g.state.to_numpy(), g.tangent.to_numpy()
                mse = np.mean(e * e)
                sc = np.mean(2 * e * s + s * delta)
                tc = np.mean(2 * e * t + t * delta)
                check = {
                    "state_attributed_error_gain": -sc / mse,
                    "tangent_attributed_error_gain": -tc / mse,
                    "total_relative_error_gain": 1 - np.mean((jj - y) ** 2) / mse,
                    "state_projection_on_change": np.dot(s, delta)
                    / np.dot(delta, delta),
                    "tangent_projection_on_change": np.dot(t, delta)
                    / np.dot(delta, delta),
                    "state_tangent_covariance": np.mean(
                        (s - s.mean()) * (t - t.mean())
                    ),
                }
                accounting_error = max(
                    accounting_error,
                    max(abs(float(effects.loc[key, k]) - v) for k, v in check.items()),
                )
    if metric_error > 1e-11 or accounting_error > 1e-11:
        raise ValueError("Independent metric/accounting audit failed")
    for stage in (pf, rf):
        if any(
            stage[n] != 0
            for n in ("fits", "native_calls", "current_target_reads", "extra_denials")
        ):
            raise ValueError("Construction dependency boundary failed")
        if (
            stage["cross_evaluations"] != 96
            or stage["diagonal_evaluations"] != 96
            or stage["denial_probes"] != 4
        ):
            raise ValueError("Exact bounded evaluation inventory failed")
    total = m["prepare_seconds"] + pf["seconds"]
    if total > m["max_seconds"] or rf["seconds"] > m["max_seconds"]:
        raise ValueError("Bounded CPU budget exceeded")
    result = {
        "status": "PASS",
        "primary_commit": m["commit"],
        "replayed_prediction_packets": packets,
        "replayed_tables": len(summaries[0]["files"]),
        "diagonal_max_error": diagonal_error,
        "metric_max_error": metric_error,
        "accounting_max_error": accounting_error,
        "primary_compute_seconds": total,
        "cached_replay_seconds": rf["seconds"],
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "query_families": len(ARMS) - 1,
        "audit_code_sha256": sha(Path(__file__)),
    }
    write(out / "replay_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    audit(BASE / "logic_response_repair_v1")
