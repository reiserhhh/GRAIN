#!/usr/bin/env python3
"""Replay and independently check the post-result whole-reader crossing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_whole_path_assembly import (
    BASE,
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
    """Check exact replay, paired units and signed accounts without neural calls."""
    verify(out)
    primary, replay = [frozen(out, n) for n in ("relocation", "relocation_replay")]
    if set(primary["files"]) != set(replay["files"]) or len(primary["files"]) != 24:
        raise ValueError("Relocation packet inventory differs")
    for record in (primary, replay):
        if (
            record["seconds"] > 60
            or record["native_calls"]
            or record["fits"]
            or record["human_labels_read"]
            or record["extra_denials"]
            or record["denial_probes"] != 4
            or record["diagonal_error"] > 1e-11
        ):
            raise ValueError("Cached crossing execution contract differs")
    for name in primary["files"]:
        a, b = [
            arrays(out / folder / name)
            for folder in ("relocation", "relocation_replay")
        ]
        if set(a) != {"cells", "geometry"} or set(a) != set(b):
            raise ValueError("Packet array inventory differs")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    summaries = [
        read(out / n / "summary.json")
        for n in ("relocation_analysis", "relocation_replay_analysis")
    ]
    for folder, r in zip(
        ("relocation_analysis", "relocation_replay_analysis"), summaries, strict=True
    ):
        verify_hashes(out / folder, r["files"])
        if (r["rows"], r["contrasts"], len(r["files"])) != (384, 192, 14):
            raise ValueError("Relocation table inventory differs")
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("Cached table replay differs")
    folder = out / "relocation_analysis"
    full = pd.read_csv(folder / "private_cells.csv")
    contrast = pd.read_csv(folder / "private_contrasts.csv")
    keys, vals = ["author_key", "cohort", "cell", "factor"], ["actual", "predicted"]
    a, b = [
        full[full.query_name == q].set_index(keys)[vals].sort_index() for q in QUERIES
    ]
    np.testing.assert_allclose(
        contrast.set_index(keys)[vals].sort_index(), a - b, atol=1e-13
    )
    metric_error, account_error = 0.0, 0.0
    for raw, stem, base_groups in (
        (full, "query", ["cohort", "query_name"]),
        (contrast, "contrast", ["cohort"]),
    ):
        for extra, suffix in (([], ""), (["factor"], "_factor")):
            groups = base_groups + extra + ["cell"]
            tab = pd.read_csv(folder / (stem + suffix + "_cells.csv")).set_index(groups)
            for key, g in raw.groupby(groups):
                y, p = g.actual.to_numpy(), g.predicted.to_numpy()
                check = {
                    "action_r": np.corrcoef(y, p)[0, 1],
                    "action_rmse": np.sqrt(np.mean((p - y) ** 2)),
                    "action_mse_over_zero": np.mean((p - y) ** 2) / np.mean(y**2),
                }
                metric_error = max(
                    metric_error,
                    *(abs(float(tab.loc[key, k]) - v) for k, v in check.items()),
                )
        e = pd.read_csv(folder / ("private_" + stem + "_effects.csv"))
        differences = (
            e.state + e.tangent - e.change,
            e.joint - e.value - e.change,
            e.joint_squared_error - e.base_squared_error - e.error_change,
            e.state_linear_error
            + e.tangent_linear_error
            + e.state_squared
            + e.tangent_squared
            + e.cross_error
            - e.error_change,
        )
        account_error = max(account_error, *(float(abs(d).max()) for d in differences))
    if metric_error > 1e-10 or account_error > 1e-10:
        raise ValueError("Independent crossing metric/accounting check failed")
    r = {
        "status": "PASS",
        "post_result_crossing_commit": primary["commit"],
        "cached_replayed_packets": 24,
        "cached_replayed_tables": 14,
        "scalar_cells": 384,
        "paired_contrasts": 192,
        "primary_seconds": primary["seconds"],
        "replay_seconds": replay["seconds"],
        "diagonal_max_error": primary["diagonal_error"],
        "independent_metric_max_error": float(metric_error),
        "independent_accounting_max_error": float(account_error),
        "audit_code_sha256": sha(Path(__file__)),
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
    }
    write(out / "relocation_replay_audit.json", r)
    print(r, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit(BASE / "logic_whole_path_assembly_v1")
