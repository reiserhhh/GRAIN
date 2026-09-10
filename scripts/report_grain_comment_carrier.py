#!/usr/bin/env python3
"""Publish a whitelist of aggregate NEXT17/18 tables after exact analysis replay."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def main():
    """Check numerical replay and publish no author, state, questionnaire or logits."""
    source = BASE / "logic_comment_carrier_v1_r2"
    first, second = source / "analysis_r1", source / "analysis_r2_replay"
    summary = read(first / "summary.json")
    if (
        summary["status"] != "COMPLETE"
        or read(second / "summary.json")["status"] != "COMPLETE"
    ):
        raise ValueError("Two complete analyses required")
    checks = {}
    for name, digest in summary["numerical_hashes"].items():
        if name.startswith("._"):
            continue
        same = digest == sha(first / name) == sha(second / name)
        if not same:
            raise ValueError("Numerical replay changed")
        checks[name] = {"sha256": digest, "byte_identical": True}
    destination = ROOT / "reports/grain_vector_mechanisms/comment_carrier_tables_v1"
    destination.mkdir(exist_ok=True)
    tables = {}
    for name in ("joint_metrics.csv", "carrier_metrics.csv", "quality.csv"):
        data = pd.read_csv(first / name)
        if set(data.columns) & {
            "author_id",
            "user_id",
            "participant_id",
            "y_true",
            "text",
            "body",
            "candidate_logits",
        }:
            raise ValueError("Private fields cannot enter public tables")
        shutil.copyfile(first / name, destination / name)
        tables[name] = {"rows": len(data), "sha256": sha(destination / name)}
    failed = read(BASE / "logic_comment_carrier_v1/run_status.json")
    if failed["calls"] != 0:
        raise ValueError("Record failed preflight cost accurately")
    write(
        destination / "reproduction.json",
        {
            "status": "COMPLETE",
            "scope": "48 exposed source authors, one view, no Y fitting",
            "protocol_commit": "90724af",
            "collector_commit": "4861095",
            "failed_preflight_calls": failed["calls"],
            "cost": summary["cost"],
            "old_endpoint_replay_max": summary["old_endpoint_replay_max"],
            "repeat_logit_error_max": summary["repeat_logit_error_max"],
            "mean_error_max": summary["mean_error_max"],
            "carrier_closure_error": summary["carrier_closure_error"],
            "interpolated_roles": summary["interpolated_roles"],
            "focused_tests": 119,
            "full_suite_run": False,
            "replayed_files": checks,
            "public_tables": tables,
            "no_neural_replay_for_analysis": True,
        },
    )
    print(json.dumps({"public_tables": tables, "replayed_files": len(checks)}))


if __name__ == "__main__":
    main()
