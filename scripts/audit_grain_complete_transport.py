#!/usr/bin/env python3
"""Audit frozen NEXT41 artifacts and retain signed branch contributions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_complete_transport_jvp import verify
from scripts.run_grain_reader_new_author_transport import verify_hashes
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Check numerical replay without another expensive native transport run."""
    out = BASE / "logic_complete_transport_jvp_v1_r2"
    verify(out)
    frozen = read(out / "prediction_freeze.json")
    verify_hashes(out, frozen["files"])
    source = read(out / "source_freeze.json")
    verify_hashes(out, source["files"])
    result = read(out / "analysis/summary.json")
    verify_hashes(ROOT, result["native_target_hashes"])
    for name, digest in result["files"].items():
        if (
            sha(out / "analysis" / name) != digest
            or sha(out / "analysis_replay" / name) != digest
        ):
            raise ValueError("Numerical analysis replay differs")
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    actions = actions[actions.channel == "response"]
    rows = []
    for keys, a in actions.groupby(["cohort", "query", "mode"], sort=True):
        d, k, f = (a[c].to_numpy() for c in ("direct", "attention", "predicted"))
        energy = float(f @ f)
        rows.append(
            dict(zip(("cohort", "query", "mode"), keys, strict=True))
            | {
                "direct_energy_over_full": float(d @ d / energy),
                "last_attention_energy_over_full": float(k @ k / energy),
                "signed_cross_over_full": float(2 * d @ k / energy),
                "opposite_signs": int(np.sum(d * k < 0)),
                "actions": len(d),
                "addition_max_error": float(abs(f - d - k).max()),
            }
        )
    branches = pd.DataFrame(rows)
    if branches.addition_max_error.max() > 1e-6:
        raise ValueError("Signed branch closure differs")
    branches.to_csv(out / "branch_accounting.csv", index=False)
    summary = {
        "prediction_packets_verified": len(frozen["files"]) // 2,
        "exact_analysis_csvs": len(result["files"]),
        "saved_contraction_replay_max": result["replay_max_error"],
        "branch_addition_max": float(branches.addition_max_error.max()),
        "source_score_replay_max": max(
            r["score_error"] for r in source["source_records"]
        ),
        "source_g_relative_l2_max": max(
            r["g_relative_l2"] for r in source["source_records"]
        ),
        "finite_differences": result["finite_differences"],
        "total_native_seconds": result["total_native_seconds"],
        "source_forwards_with_pilot": 8 + result["prior_pilot_forwards"],
        "source_backwards_with_pilot": 8 + result["prior_pilot_backwards"],
        "jvps": frozen["runtime"]["jvps"],
        "fd_calls": frozen["runtime"]["fd_calls"],
        "current_target_backwards": 0,
    }
    write(out / "replay_audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
